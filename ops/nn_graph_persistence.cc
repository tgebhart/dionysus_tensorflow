#define BOOST_EXCEPTION_DISABLE
#define BOOST_NO_EXCEPTIONS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <math.h>
#include <fstream>

#include "include/dionysus/simplex.h"
#include "include/dionysus/filtration.h"
#include "include/dionysus/ordinary-persistence.h"
#include "include/dionysus/standard-reduction.h"
#include "include/dionysus/diagram.h"
#include "include/dionysus/row-reduction.h"
#include "include/dionysus/fields/q.h"
#include "include/dionysus/fields/z2.h"

#include "wasserstein/include/wasserstein.cpp"

#include <vector>
#include <map>
#include <iostream>
#include <tuple>
#include <typeinfo>


#if 1
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#endif

using namespace tensorflow;
using namespace dionysus;

typedef std::tuple<int, int, int, int> Vertex;
typedef Simplex<Vertex, float> Smplx;
typedef Filtration<Smplx> Fltr;
typedef Z2Field Field;
typedef OrdinaryPersistenceNoNegative<Field, unsigned, std::less<unsigned>> Persistence;

REGISTER_OP("InputGraphPersistence")
    .Attr("T: list({float})")
    .Input("in_tensors: T")
    .Input("in_layers: int32")
    .Input("in_alphas: float")
    .Input("in_filename: string")
    .Output("output_tensor: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("BottleneckDistance")
    .Attr("T: list({float})")
    .Input("in_tensors: T")
    .Input("in_layers: int32")
    .Input("in_alphas: float")
    .Output("output_tensor: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("WassersteinDistance")
    .Attr("T: list({float})")
    .Input("in_tensors: T")
    .Input("in_layers: int32")
    .Input("in_alphas: float")
    .Output("output_tensor: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("LayerwisePercentile")
    .Attr("T: list({float})")
    .Input("in_tensors: T")
    .Input("in_layers: int32")
    .Input("in_percentiles: float")
    .Output("output_tensor: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

std::tuple<int,int,int,int> simpleVertex(int h, int w, int c, int l) {
  return std::make_tuple(h,w,c,l);
}

std::vector<std::tuple<int,int,int,int>> zeroVertexId(std::tuple<int,int,int,int> id) {
  std::vector<std::tuple<int,int,int,int>> v = {id};
  return v;
}

std::vector<std::tuple<int, int, int, int>> vertexId(int height, int width, int channel, int layer) {
  std::vector<std::tuple<int, int, int, int>> v = {std::make_tuple(height, width, channel, layer)};
  return v;
};

std::vector<std::tuple<int, int, int, int>> oneVertexId(int h, int w, int c, int l, int i, int x, int d, int m) {
  std::vector<std::tuple<int, int, int, int>> v = {std::make_tuple(h, w, c, l),
                                                  std::make_tuple(i, x, d, m)};
  return v;
};

void addToVertexMap(std::map<std::tuple<int,int,int,int>,float> &vertexMap, int h, int w, int c, int l, float phi) {
  std::tuple<int,int,int,int> id = simpleVertex(h, w, c, l);
  auto it = vertexMap.find(id);
  if (it != vertexMap.end()) {
    if (vertexMap[id] < phi) {
      vertexMap[id] = phi;
    };
  } else {
    vertexMap[id] = phi;
  };
};


void addConvLayer(const Tensor& input_tensor, const Tensor& filter_tensor,
                  const Tensor& output_tensor, Fltr& f, int first_layer,
                  int second_layer, int idx, float alpha,
                  std::map<std::tuple<int, int, int, int>, float>& vertexMap) {

  std::cout << "Conv: " << input_tensor.DebugString() << std::endl;

  const TensorShape& input_shape = input_tensor.shape();
  const TensorShape& output_shape = output_tensor.shape();
  const TensorShape& filter_shape = filter_tensor.shape();

  auto input = input_tensor.tensor<float,4>();
  auto filter = filter_tensor.tensor<float,4>();

  for (int ih = 0; ih < input_shape.dim_size(1); ++ih) {
    for (int iw = 0; iw < input_shape.dim_size(2); ++iw) {
      for (int ic = 0; ic < input_shape.dim_size(3); ++ic) {
        for (int oc = 0; oc < output_shape.dim_size(3); ++oc) {
          int mid_h = filter_shape.dim_size(0)/2;
          int mid_w = filter_shape.dim_size(1)/2;
          float phi = fabs(filter(mid_h,mid_w,ic,oc) * input(idx,ih,iw,ic));

          if (phi > alpha) {
            addToVertexMap(vertexMap, ih, iw, ic, first_layer, phi);
            addToVertexMap(vertexMap, ih, iw, oc, second_layer, phi);
            f.push_back(Smplx(oneVertexId(ih,iw,ic,first_layer,ih,iw,oc,second_layer), phi));
          };

          for (int h = -mid_h; h < mid_h; ++h) {
            int h_idx = ih + h;
            if (h_idx >= 0 && h_idx < input_shape.dim_size(1)) {
              for (int w = -mid_w; w < mid_w; ++w) {
                int w_idx = iw + w;
                if (w_idx >= 0 && w_idx < input_shape.dim_size(2)) {
                  if (phi > alpha) {
                    phi = fabs(filter(mid_h+h,mid_w+w,ic,oc) * input(idx,h_idx,w_idx,ic));
                    f.push_back(Smplx(oneVertexId(h_idx,w_idx,ic,first_layer,ih,iw,oc,second_layer), phi));
                  };
                };
              };
            };
          };
        };
      };
    };
  };
};


std::vector<float> convPercentile(const Tensor& input_tensor, const Tensor& filter_tensor,
                  const Tensor& output_tensor, int idx) {

  const TensorShape& input_shape = input_tensor.shape();
  const TensorShape& output_shape = output_tensor.shape();
  const TensorShape& filter_shape = filter_tensor.shape();

  std::vector<float> phis;

  auto input = input_tensor.tensor<float,4>();
  auto filter = filter_tensor.tensor<float,4>();

  for (int ih = 0; ih < input_shape.dim_size(1); ++ih) {
    for (int iw = 0; iw < input_shape.dim_size(2); ++iw) {
      for (int ic = 0; ic < input_shape.dim_size(3); ++ic) {
        for (int oc = 0; oc < output_shape.dim_size(3); ++oc) {
          int mid_h = filter_shape.dim_size(0)/2;
          int mid_w = filter_shape.dim_size(1)/2;
          float phi = fabs(filter(mid_h,mid_w,ic,oc) * input(idx,ih,iw,ic));
          phis.push_back(phi);
          for (int h = -mid_h; h < mid_h; ++h) {
            int h_idx = ih + h;
            if (h_idx >= 0 && h_idx < input_shape.dim_size(1)) {
              for (int w = -mid_w; w < mid_w; ++w) {
                int w_idx = iw + w;
                if (w_idx >= 0 && w_idx < input_shape.dim_size(2)) {
                  phi = fabs(filter(mid_h+h,mid_w+w,ic,oc) * input(idx,h_idx,w_idx,ic));
                  phis.push_back(phi);
                };
              };
            };
          };
        };
      };
    };
  };
  return phis;
};

void addFCLayerFourTwo(const Tensor& input_tensor, const Tensor& weight_tensor,
                const Tensor& output_tensor, Fltr& f, int first_layer,
                int second_layer, int idx, float alpha,
                std::map<std::tuple<int, int, int, int>, float>& vertexMap) {

  std::cout << "FC 4 x 2: " << input_tensor.DebugString() << std::endl;
  std::cout << "FC 4 x 2: " << weight_tensor.DebugString() << std::endl;

  const TensorShape& input_shape = input_tensor.shape();
  const TensorShape& weight_shape = weight_tensor.shape();

  auto input = input_tensor.tensor<float,4>();
  auto weight = weight_tensor.tensor<float,2>();

  if (input_shape.dims() <= 2) {
      std::cout << "Wrong FC Layer Shape Method!" << std::endl;
  };

  int channels = input_shape.dim_size(3);
  int hs = input_shape.dim_size(1);
  int ws = input_shape.dim_size(2);

  for (int w = 0; w < weight_shape.dim_size(1); ++w) {
    for (int c = 0; c < channels; ++c) {
      for (int fh = 0; fh < hs; ++fh) {
        for (int fw = 0; fw < ws; ++fw) {
          int channel_offset = ((ws * fh) + fw) + (ws * hs * c);
          float phi = 0.0f;
          phi = fabs(input(idx,fh,fw,c) * weight(channel_offset,w));
          if (phi > alpha) {
            addToVertexMap(vertexMap, fh, fw, c, first_layer, phi);
            addToVertexMap(vertexMap, 0, w, 0, second_layer, phi);
            f.push_back(Smplx(oneVertexId(fh,fw,c,first_layer,0,w,0,second_layer), phi));
          };
        };
      };
    };
  };
};

std::vector<float> fcFourTwoPercentile(const Tensor& input_tensor, const Tensor& weight_tensor,
                                      const Tensor& output_tensor, int idx) {

  std::vector<float> phis;

  const TensorShape& input_shape = input_tensor.shape();
  const TensorShape& weight_shape = weight_tensor.shape();

  auto input = input_tensor.tensor<float,4>();
  auto weight = weight_tensor.tensor<float,2>();

  if (input_shape.dims() <= 2) {
      std::cout << "Wrong FC Layer Shape Method!" << std::endl;
  };

  int channels = input_shape.dim_size(3);
  int hs = input_shape.dim_size(1);
  int ws = input_shape.dim_size(2);

  for (int w = 0; w < weight_shape.dim_size(1); ++w) {
    for (int c = 0; c < channels; ++c) {
      for (int fh = 0; fh < hs; ++fh) {
        for (int fw = 0; fw < ws; ++fw) {
          int channel_offset = ((ws * fh) + fw) + (ws * hs * c);
          float phi = 0.0f;
          phi = fabs(input(idx,fh,fw,c) * weight(channel_offset,w));
          phis.push_back(phi);
        };
      };
    };
  };
  return phis;
};

void addFCLayerTwoTwo(const Tensor& input_tensor, const Tensor& weight_tensor,
                const Tensor& output_tensor, Fltr& f, int first_layer,
                int second_layer, int idx, float alpha,
                std::map<std::tuple<int, int, int, int>, float>& vertexMap) {

  std::cout << "FC 2 x 2: " << input_tensor.DebugString() << std::endl;
  std::cout << "FC 2 x 2: " << weight_tensor.DebugString() << std::endl;

  const TensorShape& input_shape = input_tensor.shape();
  const TensorShape& weight_shape = weight_tensor.shape();

  auto input = input_tensor.tensor<float,2>();
  auto weight = weight_tensor.tensor<float,2>();

  if (input_shape.dims() != 2) {
    std::cout << "Wrong FC Layer Shape Method!" << std::endl;
  };

  int ws = input_shape.dim_size(1);

  for (int w = 0; w < weight_shape.dim_size(1); ++w) {
    for (int fw = 0; fw < ws; ++fw) {
      float phi = 0.0f;
      phi = fabs(input(idx,fw) * weight(fw,w));
      if (phi > alpha) {
        addToVertexMap(vertexMap, 0, fw, 0, first_layer, phi);
        addToVertexMap(vertexMap, 0, w, 0, second_layer, phi);
        f.push_back(Smplx(oneVertexId(0,fw,0,first_layer,0,w,0,second_layer), phi));
      };
    };
  };
};

std::vector<float> fcTwoTwoPercentile(const Tensor& input_tensor, const Tensor& weight_tensor,
                                      const Tensor& output_tensor, int idx) {

  std::vector<float> phis;

  const TensorShape& input_shape = input_tensor.shape();
  const TensorShape& weight_shape = weight_tensor.shape();

  auto input = input_tensor.tensor<float,2>();
  auto weight = weight_tensor.tensor<float,2>();

  if (input_shape.dims() != 2) {
    std::cout << "Wrong FC Layer Shape Method!" << std::endl;
  };

  int ws = input_shape.dim_size(1);

  for (int w = 0; w < weight_shape.dim_size(1); ++w) {
    for (int fw = 0; fw < ws; ++fw) {
      float phi = 0.0f;
      phi = fabs(input(idx,fw) * weight(fw,w));
      phis.push_back(phi);
    };
  };
  return phis;
};

float computePercentile(std::vector<float> phis, float percentile) {
  phis.erase(std::remove(phis.begin(), phis.end(), 0), phis.end());
  std::nth_element(phis.begin(), phis.begin() + (percentile*phis.size())/100.0, phis.end());
  std::cout << "Percentile: " << phis[(percentile*phis.size())/100.0] << std::endl;
  return phis[(percentile*phis.size())/100.0];
};

void addVertexMapToFilter(std::map<std::tuple<int, int, int, int>, float>& vertexMap,
                          Fltr &f) {
  for (auto &spx : vertexMap) {
    f.push_back(Smplx(zeroVertexId(spx.first), spx.second));
  }
};

class InputGraphPersistenceOp : public OpKernel
{
  public:
    explicit InputGraphPersistenceOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {

      int num_inputs = context->num_inputs();

      const Tensor& l_input = context->input(num_inputs-3);
      auto layers = l_input.flat<int>();

      const Tensor& a_input = context->input(num_inputs-2);
      auto alphas = a_input.tensor<float, 2>();

      const Tensor& f_input = context->input(num_inputs-1);
      auto f_tensor = f_input.flat<std::string>();
      std::string filename = f_tensor(0);

      Fltr f;
      std::map<std::tuple<int, int, int, int>, float> simplexMap;
      int numLayers = layers.dimension(0);
      int layerType;
      int layer = 0;
      for (int i = 0; i < numLayers; ++i) {
        if (i % (3) == 2) {
          layerType = layers(i);
          if (layerType == 2) {
            addConvLayer(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(0,layer), simplexMap);
            std::cout << "Added conv layer" << std::endl;
          };
          if (layerType == 4) {
            if (layers(i-2) == 2 || layers(i-2) == 0) {
                addFCLayerFourTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(0,layer), simplexMap);
                std::cout << "Added FC 4 x 2 layer" << std::endl;
            } else {
                addFCLayerTwoTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(0,layer), simplexMap);
                std::cout << "Added FC 2 x 2 layer" << std::endl;
            };
          };
          layer++;
        };
      };

      addVertexMapToFilter(simplexMap, f);
      Field q;


      // std::ofstream debugFile;
      // debugFile.open("/home/tgebhart/python/projects/tf_activation/logdir/data/filtration.csv");
      // for (auto& s : f)
      // {
      //     debugFile << s.size() << "," << s.data() << "\n";
      // }
      std::cout << "Filtration initialized" << std::endl;
      f.sort(DataDimensionComparisonReverse<Smplx>());
      std::cout << "filtration size: " << f.size() << std::endl;

      Persistence persistence(q);
      StandardReduction<Persistence> reduce(persistence);
      reduce(f);
      std::cout << "Persistence initialized" << std::endl;

      auto diagrams = init_diagrams(persistence, f, [&](const Smplx& s) -> float { return f.index(s); },
                    [](Persistence::Index i) { return i; });

      std::ofstream outputFile;
      outputFile.open(filename.c_str());
      for (auto& pt : diagrams[0]) {
        //std::cout << "birth: " << f[pt.birth()].data() << " death: " << f[pt.death()].data() << " " << pt.data << std::endl;
        outputFile << f[pt.birth()].data() << "," << f[pt.death()].data() << "\n";
        // outputFile << pt.birth() << "," << pt.death() << "\n";
      };
      // std::cout << "Diagram H_0: " << dgms[0] << std::endl;

      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, l_input.shape(),
                                                     &output_tensor));
    };
};


class BottleneckDistanceOp : public OpKernel
{
  public:
    explicit BottleneckDistanceOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {

      int num_inputs = context->num_inputs();

      const Tensor& l_input = context->input(num_inputs-2);
      auto layers = l_input.flat<int>();

      const Tensor& a_input = context->input(num_inputs-1);
      auto alphas = a_input.tensor<float, 2>();

      Fltr ff;
      std::map<std::tuple<int, int, int, int>, float> simplexMapF;


      int numLayers = layers.dimension(0);
      int layerType;
      int layer = 0;
      for (int i = 0; i < numLayers; ++i) {
        if (i % (3) == 2) {
          layerType = layers(i);
          if (layerType == 2) {
            addConvLayer(context->input(i-2), context->input(i-1), context->input(i), ff, layer, layer+1, 0, alphas(0,layer), simplexMapF);
            std::cout << "Added conv layer" << std::endl;
          };
          if (layerType == 4) {
            if (layers(i-2) == 2 || layers(i-2) == 0) {
                addFCLayerFourTwo(context->input(i-2), context->input(i-1), context->input(i), ff, layer, layer+1, 0, alphas(0,layer), simplexMapF);
                std::cout << "Added FC 4 x 2 layer" << std::endl;
            } else {
                addFCLayerTwoTwo(context->input(i-2), context->input(i-1), context->input(i), ff, layer, layer+1, 0, alphas(0,layer), simplexMapF);
                std::cout << "Added FC 2 x 2 layer" << std::endl;
            };
          };
          layer++;
        };
      };
      addVertexMapToFilter(simplexMapF, ff);

      Fltr fs;
      std::map<std::tuple<int, int, int, int>, float> simplexMapS;
      layer = 0;
      for (int i = 0; i < numLayers; ++i) {
        if (i % (3) == 2) {
          layerType = layers(i);
          if (layerType == 2) {
            addConvLayer(context->input(i-2), context->input(i-1), context->input(i), fs, layer, layer+1, 1, alphas(1,layer), simplexMapS);
            std::cout << "Added conv layer" << std::endl;
          };
          if (layerType == 4) {
            if (layers(i-2) == 2 || layers(i-2) == 0) {
                addFCLayerFourTwo(context->input(i-2), context->input(i-1), context->input(i), fs, layer, layer+1, 1, alphas(1,layer), simplexMapS);
                std::cout << "Added FC 4 x 2 layer" << std::endl;
            } else {
                addFCLayerTwoTwo(context->input(i-2), context->input(i-1), context->input(i), fs, layer, layer+1, 1, alphas(1,layer), simplexMapS);
                std::cout << "Added FC 2 x 2 layer" << std::endl;
            };
          };
          layer++;
        };
      };
      addVertexMapToFilter(simplexMapS, fs);

      Field q;

      ff.sort(DataDimensionComparisonReverse<Smplx>());
      fs.sort(DataDimensionComparisonReverse<Smplx>());
      std::cout << "First Filtration initialized" << std::endl;
      std::cout << "Size of first filration: " << ff.size() << std::endl;

      Persistence persistence(q);
      StandardReduction<Persistence> reduce(persistence);
      reduce(ff);
      std::cout << "Persistence initialized" << std::endl;

      auto dgmsf = init_diagrams(persistence, ff, [&](const Smplx& s) -> float {  return ff.index(s); },
                    [](Persistence::Index i) { return i; });

      Persistence persistenceS(q);
      StandardReduction<Persistence> reduceS(persistenceS);
      reduceS(fs);
      std::cout << "Second Persistence initialized" << std::endl;
      std::cout << "Size of second filration: " << fs.size() << std::endl;

      auto dgmss = init_diagrams(persistenceS, fs, [&](const Smplx& s) -> float {  return fs.index(s); },
                   [](Persistence::Index i) { return i; });

      std::cout << "Computing Bottleneck Distance via Hera" << std::endl;
      // double bottleneck = geom_bt::bottleneckDistApprox(dgmsf[0], dgmss[0], 0.1);
      std::cout << "Finished Hera Bottleneck computation" << std::endl;

      double bottleneck = 0;

      TensorShape output_shape;
      output_shape.AddDim(1);
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

      auto output_flat = output_tensor->flat<float>();
      output_flat(0) = bottleneck;

    }
};

class WassersteinDistanceOp : public OpKernel
{
  public:
    explicit WassersteinDistanceOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {

      int num_inputs = context->num_inputs();

      const Tensor& l_input = context->input(num_inputs-2);
      auto layers = l_input.flat<int>();

      const Tensor& a_input = context->input(num_inputs-1);
      auto alphas = a_input.tensor<float, 2>();

      Fltr ff;
      std::map<std::tuple<int, int, int, int>, float> simplexMapF;

      int numLayers = layers.dimension(0);
      int layerType;
      int layer = 0;
      for (int i = 0; i < numLayers; ++i) {
        if (i % (3) == 2) {
          layerType = layers(i);
          if (layerType == 2) {
            addConvLayer(context->input(i-2), context->input(i-1), context->input(i), ff, layer, layer+1, 0, alphas(0,layer), simplexMapF);
            std::cout << "Added conv layer" << std::endl;
          };
          if (layerType == 4) {
            if (layers(i-2) == 2 || layers(i-2) == 0) {
                addFCLayerFourTwo(context->input(i-2), context->input(i-1), context->input(i), ff, layer, layer+1, 0, alphas(0,layer), simplexMapF);
                std::cout << "Added FC 4 x 2 layer" << std::endl;
            } else {
                addFCLayerTwoTwo(context->input(i-2), context->input(i-1), context->input(i), ff, layer, layer+1, 0, alphas(0,layer), simplexMapF);
                std::cout << "Added FC 2 x 2 layer" << std::endl;
            };
          };
          layer++;
        };
      };
      addVertexMapToFilter(simplexMapF, ff);

      Fltr fs;
      std::map<std::tuple<int, int, int, int>, float> simplexMapS;
      layer = 0;
      for (int i = 0; i < numLayers; ++i) {
        if (i % (3) == 2) {
          layerType = layers(i);
          if (layerType == 2) {
            addConvLayer(context->input(i-2), context->input(i-1), context->input(i), fs, layer, layer+1, 1, alphas(1,layer), simplexMapS);
            std::cout << "Added conv layer" << std::endl;
          };
          if (layerType == 4) {
            if (layers(i-2) == 2 || layers(i-2) == 0) {
                addFCLayerFourTwo(context->input(i-2), context->input(i-1), context->input(i), fs, layer, layer+1, 1, alphas(1,layer), simplexMapS);
                std::cout << "Added FC 4 x 2 layer" << std::endl;
            } else {
                addFCLayerTwoTwo(context->input(i-2), context->input(i-1), context->input(i), fs, layer, layer+1, 1, alphas(1,layer), simplexMapS);
                std::cout << "Added FC 2 x 2 layer" << std::endl;
            };
          };
          layer++;
        };
      };
      addVertexMapToFilter(simplexMapS, fs);

      Field q;
      Field q2;

      ff.sort(DataDimensionComparisonReverse<Smplx>());
      fs.sort(DataDimensionComparisonReverse<Smplx>());
      std::ofstream debugFile;
      // debugFile.open("/home/tgebhart/python/projects/tf_activation/logdir/data/filtration.csv");
      // for (auto& s : fs)
      // {
      //     debugFile << s.size() << "," << s.data() << "\n";
      // }
      std::cout << "First Filtration initialized" << std::endl;
      std::cout << "Size of first filration: " << ff.size() << std::endl;

      Persistence persistence(q);
      StandardReduction<Persistence> reduce(persistence);
      reduce(ff);
      std::cout << "Persistence initialized" << std::endl;

      auto dgmsf = init_diagrams(persistence, ff, [&](const Smplx& s) -> float {  return ff.index(s); },
                    [](Persistence::Index i) { return i; });

      std::cout << "Beginning second persistence" << std::endl;
      Persistence persistenceS(q2);
      std::cout << "Second persistence initialize" << std::endl;
      StandardReduction<Persistence> reduceS(persistenceS);
      std::cout << "reducing fs" << std::endl;
      std::cout << "Size of second filration: " << fs.size() << std::endl;
      reduceS(fs);
      std::cout << "Second Persistence initialized" << std::endl;

      auto dgmss = init_diagrams(persistenceS, fs, [&](const Smplx& s) -> float {  return fs.index(s); },
                   [](Persistence::Index i) { return i; });

      // std::ofstream outputFile;
      // outputFile.open("/home/tgebhart/python/projects/tf_activation/logdir/data/wass1.csv");

      // std::vector<int> removeF;
      // std::vector<int> removeS;
      // int i = 0;
      // for (auto& pt : dgmsf[0]) {
      //   // std::cout << "birth: " << f[pt.birth()].data() << " death: " << f[pt.death()].data() << " " << pt.data << std::endl;
      //   // auto temp = pt.birth();
      //   // pt.setBirth(pt.death());
      //   // pt.setDeath(temp);
      //   if (ff[pt.birth()].data() > ff[pt.death()].data()) {
      //        outputFile << ff[pt.birth()].data() << "," << ff[pt.death()].data() << std::endl;
      //   } else {
      //    removeF.push_back(i);
      //  }
      //  i++;
      // };
      //
      // std::ofstream outputFileTwo;
      // outputFileTwo.open("/home/tgebhart/python/projects/tf_activation/logdir/data/wass2.csv");
      // i = 0;
      // for (auto& pt : dgmss[0]) {
      //   // auto temp = pt.birth();
      //   // pt.setBirth(pt.death());
      //   // pt.setDeath(temp);
      //   if (fs[pt.birth()].data() > fs[pt.death()].data()) {
      //   //   std::cout << "birth: " << fs[pt.birth()].data() << " death: " << fs[pt.death()].data() << " " << pt.data << std::endl;
      //     outputFileTwo << fs[pt.birth()].data() << "," << fs[pt.death()].data() << std::endl;
      //   } else {
      //    removeS.push_back(i);
      //  }
      //  i++;
      // };
      //
      // dgmsf[0].delete_points(removeF);
      // dgmss[0].delete_points(removeS);

      std::cout << "Computing Wasserstein Distance via Hera" << std::endl;
      double wasserstein = geom_ws::wassersteinDist(dgmsf[0], dgmss[0], 2.0, 0.5);
      std::cout << "Finished Hera wasserstein_distance computation" << std::endl;

      TensorShape output_shape;
      output_shape.AddDim(1);
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

      auto output_flat = output_tensor->flat<float>();
      output_flat(0) = wasserstein;
    };
};

class LayerwisePercentileOp : public OpKernel
{
  public:
    explicit LayerwisePercentileOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {

      int num_inputs = context->num_inputs();

      const Tensor& l_input = context->input(num_inputs-2);
      auto layers = l_input.flat<int>();

      const Tensor& p_input = context->input(num_inputs-1);
      auto percentiles = p_input.flat<float>();

      int numLayers = layers.dimension(0);
      int layerType;
      int layer = 0;

      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, p_input.shape(),
                                                     &output_tensor));
      auto output_flat = output_tensor->flat<float>();

      std::vector<float> phis;

      for (int i = 0; i < numLayers; ++i) {
        if (i % (3) == 2) {
          layerType = layers(i);
          if (layerType == 2) {
            phis = convPercentile(context->input(i-2), context->input(i-1), context->input(i), 0);
            output_flat(layer) = computePercentile(phis, percentiles(layer));
          };
          if (layerType == 4) {
            if (layers(i-2) == 2 || layers(i-2) == 0) {
                phis = fcFourTwoPercentile(context->input(i-2), context->input(i-1), context->input(i), 0);
                output_flat(layer) = computePercentile(phis, percentiles(layer));
            } else {
                phis = fcTwoTwoPercentile(context->input(i-2), context->input(i-1), context->input(i), 0);
                output_flat(layer) = computePercentile(phis, percentiles(layer));
            };
          };
          layer++;
        };
      };
    };
};




REGISTER_KERNEL_BUILDER(Name("InputGraphPersistence").Device(DEVICE_CPU), InputGraphPersistenceOp);
REGISTER_KERNEL_BUILDER(Name("BottleneckDistance").Device(DEVICE_CPU), BottleneckDistanceOp);
REGISTER_KERNEL_BUILDER(Name("WassersteinDistance").Device(DEVICE_CPU), WassersteinDistanceOp);
REGISTER_KERNEL_BUILDER(Name("LayerwisePercentile").Device(DEVICE_CPU), LayerwisePercentileOp);
