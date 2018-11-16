// #define BOOST_EXCEPTION_DISABLE
// #define BOOST_NO_EXCEPTIONS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <math.h>
#include <fstream>

#include "include/dionysus/simplex.h"
#include "include/dionysus/filtration.h"
#include "include/dionysus/ordinary-persistence.h"
#include "include/dionysus/pair-recorder.h"
#include "include/dionysus/cohomology-persistence.h"
#include "include/dionysus/standard-reduction.h"
#include "include/dionysus/reduction.h"
#include "include/dionysus/diagram.h"
#include "include/dionysus/row-reduction.h"
#include "include/dionysus/fields/q.h"
#include "include/dionysus/fields/z2.h"
#include "include/dionysus/fields/zp.h"

#include "wasserstein/include/wasserstein.h"

#include <vector>
#include <map>
#include <iostream>
#include <tuple>
#include <typeinfo>
#include <stdexcept>

using namespace boost;

#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphml.hpp>
#include <boost/throw_exception.hpp>
#include <boost/exception/exception.hpp>
#include <boost/lexical_cast.hpp>

using namespace tensorflow;
using namespace dionysus;

typedef std::vector<int> Vertex;
typedef Simplex<Vertex, float> Smplx;
typedef Filtration<Smplx> Fltr;
// typedef Z2Field Field;
// typedef ZpField<11> Field;
typedef Q<> Field;
typedef OrdinaryPersistenceNoNegative<Field, unsigned, std::less<unsigned>> Persistence;
typedef PairChainRecorder<CohomologyPersistence<Field, unsigned, std::less<unsigned>>> CoPersistence;
typedef Diagram<float, unsigned> Diag;
//typedef Diag::Points DiagPoints;

struct vertex_prop {
  std::string vertex_id;
};

typedef std::pair<Vertex, Vertex> Edge;
typedef boost::property<boost::edge_weight_t, float> EdgeWeightProperty;
typedef boost::property<boost::vertex_index_t, Vertex> VertexIndexProperty;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, vertex_prop, EdgeWeightProperty> Graph;
// typedef boost::labeled_graph<AdjList, Vertex, boost::hash_mapS> Graph;


REGISTER_OP("InputGraphPersistence")
    .Attr("T: list({float})")
    .Attr("O: {float} = DT_FLOAT")
    .Input("in_tensors: T")
    .Input("in_layers: int32")
    .Input("in_alphas: float")
    .Input("in_h_dim: int32")
    .Input("in_filename: string")
    .Output("output_tensor: O")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("InputGraphCoPersistence")
    .Attr("T: list({float})")
    .Attr("O: {float, float, float} = DT_FLOAT")
    .Input("in_tensors: T")
    .Input("in_layers: int32")
    .Input("in_alphas: float")
    .Input("in_h_dim: int32")
    .Input("in_filename: string")
    .Output("output_tensor: O")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("InputGraphFiltration")
    .Attr("T: list({float})")
    .Attr("O: {float} = DT_FLOAT")
    .Input("in_tensors: T")
    .Input("in_layers: int32")
    .Input("in_alphas: float")
    .Input("in_filename: string")
    .Output("output_tensor: O")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("PersistentSubGraph")
    .Attr("T: list({float})")
    .Attr("O: {float, float} = DT_FLOAT")
    .Input("in_tensors: T")
    .Input("in_layers: int32")
    .Input("in_alphas: float")
    .Input("in_percentile: float")
    .Input("in_h_dim: int32")
    .Input("in_filename: string")
    .Output("output_tensor: O")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_OP("WassersteinDistance")
    .Attr("T: list({float})")
    .Input("in_tensors: T")
    .Input("in_layers: int32")
    .Input("in_alphas: float")
    .Input("in_h_dim: int32")
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

Vertex simpleVertex(int h, int w, int c, int l) {
  Vertex ret = {h,w,c,l};
  return ret;
}

std::vector<Vertex> zeroVertexId(Vertex id) {
  std::vector<Vertex> v = {id};
  return v;
}

std::vector<Vertex> vertexId(int height, int width, int channel, int layer) {
  Vertex in = {height, width, channel, layer};
  std::vector<Vertex> v = {in};
  return v;
};

std::vector<Vertex> oneVertexId(int h, int w, int c, int l, int i, int x, int d, int m) {
  Vertex one = {h,w,c,l};
  Vertex two = {i,x,d,m};
  std::vector<Vertex> v = {one,two};
  return v;
};


std::vector<Vertex> twoVertexId(int h, int w, int c, int l, int i, int x, int d, int m, int p, int q, int r, int s) {
  Vertex one = {h,w,c,l};
  Vertex two = {i,x,d,m};
  Vertex three = {p,q,r,s};
  std::vector<Vertex> v = {one,two,three};
  return v;
};


void addToVertexMap(std::map<Vertex,float> &vertexMap, int h, int w, int c, int l, float phi) {
  Vertex id = simpleVertex(h, w, c, l);
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
                  std::map<Vertex,float>& vertexMap) {

  std::cout << "Conv input tensor: " << input_tensor.DebugString() << std::endl;
  std::cout << "filter tensor: " << filter_tensor.DebugString() << std::endl;
  std::cout << "output tensor: " << output_tensor.DebugString() << std::endl;

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
          float phi = 0.0f;
          phi = fabs(filter(mid_h,mid_w,ic,oc));

          if (phi > alpha) {
            addToVertexMap(vertexMap, ih, iw, ic, first_layer, phi);
            addToVertexMap(vertexMap, ih, iw, oc, second_layer, phi);
            f.push_back(Smplx(oneVertexId(ih,iw,ic,first_layer,ih,iw,oc,second_layer), phi));
          };
          float old_phi = 0.0f;
          old_phi = phi;
          for (int h = -mid_h; h < mid_h; ++h) {
            int h_idx = ih + h;
            if (h_idx >= 0 && h_idx < input_shape.dim_size(1)) {
              for (int w = -mid_w; w < mid_w; ++w) {
                int w_idx = iw + w;
                if (w_idx >= 0 && w_idx < input_shape.dim_size(2)) {
                  phi = fabs(filter(mid_h+h,mid_w+w,ic,oc));
                  if (phi > alpha) {
                    addToVertexMap(vertexMap, h_idx, w_idx, ic, first_layer, phi);
                    f.push_back(Smplx(oneVertexId(h_idx,w_idx,ic,first_layer,ih,iw,oc,second_layer), phi));
                    if (old_phi < phi) {
                      addToVertexMap(vertexMap, ih, iw, oc, second_layer, phi);
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
          float phi = 0.0f;
          phi = fabs(filter(mid_h,mid_w,ic,oc));
          phis.push_back(phi);
          for (int h = -mid_h; h < mid_h; ++h) {
            int h_idx = ih + h;
            if (h_idx >= 0 && h_idx < input_shape.dim_size(1)) {
              for (int w = -mid_w; w < mid_w; ++w) {
                int w_idx = iw + w;
                if (w_idx >= 0 && w_idx < input_shape.dim_size(2)) {
                  phi = fabs(filter(mid_h+h,mid_w+w,ic,oc));
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
                std::map<Vertex,float>& vertexMap) {

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
          phi = fabs(weight(channel_offset,w));
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
          phi = fabs(weight(channel_offset,w));
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
                std::map<Vertex,float>& vertexMap) {

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
      phi = fabs(weight(fw,w));
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
      phi = fabs(weight(fw,w));
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

void addVertexMapToFilter(std::map<Vertex,float>& vertexMap, Fltr &f) {
  for (auto &spx : vertexMap) {
    f.push_back(Smplx(zeroVertexId(spx.first), spx.second));
  }
};

int computePercentilePersistenceLifeIndex(float percentile, Diag &d, Fltr &f) {
  std::vector<int> nzeros;
  for (auto& pt : d) {
    if (pt.death() - pt.birth() > 1) {
      nzeros.push_back(pt.death() - pt.birth());
    };
  };
  std::sort(nzeros.begin(), nzeros.end());
  return nzeros[percentile*nzeros.size()/100];
};

float computePercentilePersistenceLife(float percentile, Diag &d, Fltr &f) {
  std::vector<float> nzeros;
  for (auto& pt : d) {
    float b = f[pt.birth()].data();
    float d = f[pt.death()].data();
    if (b-d > 0) {
      nzeros.push_back(b-d);
    };
  };
  return computePercentile(nzeros, percentile);
};

std::string vertexToString(Vertex v) {
  std::string word;
  for (int i = 0; i < v.size(); ++i) {
    if(i != 0)
      word += ",";
    word += std::to_string(v[i]);
  };
  return word;
};

void buildSubgraph(int idx, Fltr &f, Graph &g, int l_bound, int u_bound, Graph::vertex_descriptor u) {
  for (int i = l_bound; i <= u_bound; ++i) {
    if (f[i].dimension() > 0) {
      if (f[idx] == f[i].boundary()[0]) {
        Graph::vertex_descriptor v = boost::add_vertex(g);
        g[v].vertex_id = vertexToString(f[i].boundary()[1][0]);
        boost::add_edge(u,v,f[i].data(),g);
        buildSubgraph(f.index(f[i].boundary()[1]),f,g,i+1,u_bound,v);
      };
      if (f[idx] == f[i].boundary()[1]) {
        Graph::vertex_descriptor v = boost::add_vertex(g);
        g[v].vertex_id = vertexToString(f[i].boundary()[0][0]);
        boost::add_edge(u,v,f[i].data(),g);
        buildSubgraph(f.index(f[i].boundary()[0]),f,g,i+1,u_bound,v);
      };
    };
  };
};

class InputGraphPersistenceOp : public OpKernel
{
  public:
    explicit InputGraphPersistenceOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {

      int num_inputs = context->num_inputs();

      const Tensor& l_input = context->input(num_inputs-4);
      auto layers = l_input.flat<int>();

      const Tensor& a_input = context->input(num_inputs-3);
      auto alphas = a_input.flat<float>();

      const Tensor& h_dim = context->input(num_inputs-2);
      auto homology_dimension_tensor = h_dim.flat<int>();
      int homology_dimension = homology_dimension_tensor(0);

      const Tensor& f_input = context->input(num_inputs-1);
      auto f_tensor = f_input.flat<std::string>();
      std::string filename = f_tensor(0);

      Fltr f;
      std::map<Vertex,float> simplexMap;
      int numLayers = layers.dimension(0);
      int layerType;
      int layer = 0;
      for (int i = 0; i < numLayers; ++i) {
        if (i % (3) == 2) {
          layerType = layers(i);
          if (layerType == 2) {
            addConvLayer(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
            std::cout << "Added conv layer" << std::endl;
          };
          if (layerType == 4) {
            if (layers(i-2) == 2 || layers(i-2) == 0) {
                addFCLayerFourTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
                std::cout << "Added FC 4 x 2 layer" << std::endl;
            } else {
                addFCLayerTwoTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
                std::cout << "Added FC 2 x 2 layer" << std::endl;
            };
          };
          layer++;
        };
      };

      addVertexMapToFilter(simplexMap, f);
      Field q;

      std::ofstream debugFile;
      debugFile.open("/home/gebha095/projects/tf_activation/logdir/data/filtration.csv");

      std::cout << "Filtration initialized" << std::endl;
      f.sort(DataDimensionComparisonReverse<Smplx>());
      std::cout << "filtration size: " << f.size() << std::endl;

      // for (auto& p : f) {
      //   debugFile << p << "," << p.data() << std::endl;
      // }

      // Persistence persistence(q);
      // StandardReduction<Persistence> reduce(persistence);
      // reduce(f);
      //
      // std::cout << "Persistence initialized" << std::endl;
      //
      // auto diagrams = init_diagrams(persistence, f, [&](const Smplx& s) -> float { return f.index(s); },
      //               [](Persistence::Index i) { return i; });

      std::cout << "Computed Diagrams" << filename.c_str() << std::endl;
      std::ofstream outputFile;
      outputFile.open(filename.c_str());
      // std::cout << "diagram size: " << diagrams[homology_dimension].size() << std::endl;

      TensorShape output_shape;
      output_shape.AddDim(f.size());
      // output_shape.AddDim(homology_dimension+2);
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

      auto output = output_tensor->tensor<float,1>();
      // int i = 0;
      // for (auto& pt : diagrams[homology_dimension]) {
      //   outputFile << f[pt.birth()].data() << "," << f[pt.death()].data() << "," << pt.birth() << "," << pt.death() << "\n";
      //   i++;
      // };
      for (int i = 0; i < f.size(); ++i) {
        output(i) = f[i].data();
      };
    };
};


class InputGraphCoPersistenceOp : public OpKernel
{
  public:
    explicit InputGraphCoPersistenceOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {

      int num_inputs = context->num_inputs();

      const Tensor& l_input = context->input(num_inputs-4);
      auto layers = l_input.flat<int>();

      const Tensor& a_input = context->input(num_inputs-3);
      auto alphas = a_input.flat<float>();

      const Tensor& h_dim = context->input(num_inputs-2);
      auto homology_dimension_tensor = h_dim.flat<int>();
      int homology_dimension = homology_dimension_tensor(0);

      const Tensor& f_input = context->input(num_inputs-1);
      auto f_tensor = f_input.flat<std::string>();
      std::string filename = f_tensor(0);

      Fltr f;
      std::map<Vertex,float> simplexMap;
      int numLayers = layers.dimension(0);
      int layerType;
      int layer = 0;
      for (int i = 0; i < numLayers; ++i) {
        if (i % (3) == 2) {
          layerType = layers(i);
          if (layerType == 2) {
            addConvLayer(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
            std::cout << "Added conv layer" << std::endl;
          };
          if (layerType == 4) {
            if (layers(i-2) == 2 || layers(i-2) == 0) {
                addFCLayerFourTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
                std::cout << "Added FC 4 x 2 layer" << std::endl;
            } else {
                addFCLayerTwoTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
                std::cout << "Added FC 2 x 2 layer" << std::endl;
            };
          };
          layer++;
        };
      };

      addVertexMapToFilter(simplexMap, f);
      Field q;

      std::ofstream debugFile;
      debugFile.open("/home/gebha095/projects/tf_activation/logdir/data/filtration.csv");

      std::cout << "Filtration initialized" << std::endl;
      f.sort(DataDimensionComparisonReverse<Smplx>());
      std::cout << "filtration size: " << f.size() << std::endl;


      CoPersistence cohomology_persistence(q);
      StandardReduction<CoPersistence> reduce(cohomology_persistence);
      reduce(f);

      std::cout << "Persistence initialized" << std::endl;

      auto diagrams = init_diagrams(cohomology_persistence, f, [&](const Smplx& s) -> float { return f.index(s); },
                    [](CoPersistence::Index i) { return i; });

      std::cout << "Computed Diagrams" << filename.c_str() << std::endl;
      std::ofstream outputFile;
      outputFile.open(filename.c_str());
      std::cout << "diagram size: " << diagrams[homology_dimension].size() << std::endl;

      TensorShape output_shape;
      output_shape.AddDim(diagrams[homology_dimension].size());
      output_shape.AddDim(2);
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

      auto output = output_tensor->tensor<float,2>();
      int i = 0;
      for (auto& pt : diagrams[homology_dimension]) {
        outputFile << f[pt.birth()].data() << "," << f[pt.death()].data() << "\n";
	      output(i,0) = f[pt.birth()].data();
	      output(i,1) = f[pt.death()].data();
        i++;
      };
    };
};


class InputGraphFiltrationOp : public OpKernel
{
  public:
    explicit InputGraphFiltrationOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {

      int num_inputs = context->num_inputs();

      const Tensor& l_input = context->input(num_inputs-3);
      auto layers = l_input.flat<int>();

      const Tensor& a_input = context->input(num_inputs-2);
      auto alphas = a_input.flat<float>();

      const Tensor& f_input = context->input(num_inputs-1);
      auto f_tensor = f_input.flat<std::string>();
      std::string filename = f_tensor(0);

      Fltr f;
      std::map<Vertex,float> simplexMap;
      int numLayers = layers.dimension(0);
      int layerType;
      int layer = 0;
      for (int i = 0; i < numLayers; ++i) {
        if (i % (3) == 2) {
          layerType = layers(i);
          if (layerType == 2) {
            addConvLayer(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
            std::cout << "Added conv layer" << std::endl;
          };
          if (layerType == 4) {
            if (layers(i-2) == 2 || layers(i-2) == 0) {
                addFCLayerFourTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
                std::cout << "Added FC 4 x 2 layer" << std::endl;
            } else {
                addFCLayerTwoTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
                std::cout << "Added FC 2 x 2 layer" << std::endl;
            };
          };
          layer++;
        };
      };

      addVertexMapToFilter(simplexMap, f);
      Field q;

      std::ofstream debugFile;
      debugFile.open("/home/gebha095/projects/tf_activation/logdir/data/filtration.csv");

      std::cout << "Filtration initialized" << std::endl;
      f.sort(DataDimensionComparisonReverse<Smplx>());
      std::cout << "filtration size: " << f.size() << std::endl;

      std::cout << "Computed Diagrams" << filename.c_str() << std::endl;
      std::ofstream outputFile;
      outputFile.open(filename.c_str());

      TensorShape output_shape;
      output_shape.AddDim(f.size());
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

      auto output = output_tensor->tensor<float,1>();
      // int i = 0;
      // for (auto& pt : diagrams[homology_dimension]) {
      //   outputFile << f[pt.birth()].data() << "," << f[pt.death()].data() << "," << pt.birth() << "," << pt.death() << "\n";
      //   i++;
      // };
      for (int i = 0; i < f.size(); ++i) {
        output(i) = f[i].data();
      };
    };
};


class PersistentSubGraphOp : public OpKernel
{
  public:
    explicit PersistentSubGraphOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {

      int num_inputs = context->num_inputs();

      const Tensor& l_input = context->input(num_inputs-5);
      auto layers = l_input.flat<int>();

      const Tensor& a_input = context->input(num_inputs-4);
      auto alphas = a_input.flat<float>();

      const Tensor& percentile_input = context->input(num_inputs-3);
      auto percentile_tensor = percentile_input.flat<float>();
      float percentile = percentile_tensor(0);

      const Tensor& h_dim = context->input(num_inputs-2);
      auto homology_dimension_tensor = h_dim.flat<int>();
      int homology_dimension = homology_dimension_tensor(0);

      const Tensor& f_input = context->input(num_inputs-1);
      auto f_tensor = f_input.flat<std::string>();
      std::string filename = f_tensor(0);

      Fltr f;
      std::map<Vertex, float> simplexMap;
      int numLayers = layers.dimension(0);
      int layerType;
      int layer = 0;
      for (int i = 0; i < numLayers; ++i) {
        if (i % (3) == 2) {
          layerType = layers(i);
          if (layerType == 2) {
            addConvLayer(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
            std::cout << "Added conv layer" << std::endl;
          };
          if (layerType == 4) {
            if (layers(i-2) == 2 || layers(i-2) == 0) {
                addFCLayerFourTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
                std::cout << "Added FC 4 x 2 layer" << std::endl;
            } else {
                addFCLayerTwoTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0, alphas(layer), simplexMap);
                std::cout << "Added FC 2 x 2 layer" << std::endl;
            };
          };
          layer++;
        };
      };

      addVertexMapToFilter(simplexMap, f);
      Field q;

      std::cout << "Filtration initialized" << std::endl;
      f.sort(DataDimensionComparisonReverse<Smplx>());
      std::cout << "filtration size: " << f.size() << std::endl;

      Persistence persistence(q);
      StandardReduction<Persistence> reduce(persistence);
      reduce(f);
      std::cout << "Persistence initialized" << std::endl;

      auto diagrams = init_diagrams(persistence, f, [&](const Smplx& s) -> float { return f.index(s); },
                    [](Persistence::Index i) { return i; });

      std::cout << "diagram size: " << diagrams[homology_dimension].size() << std::endl;
      std::ofstream outputFile;
      outputFile.open(filename.c_str());

      TensorShape output_shape;
      output_shape.AddDim(diagrams[homology_dimension].size());
      output_shape.AddDim(2);
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

      Graph g;

      float p_cutoff = computePercentilePersistenceLife(percentile, diagrams[homology_dimension], f);
      std::cout << "Percentile Cutoff: " << p_cutoff << std::endl;
      auto output = output_tensor->tensor<float,2>();
      int i = 0;
      for (auto& pt : diagrams[homology_dimension]) {
        if (f[pt.birth()].data() - f[pt.death()].data() > p_cutoff && f[pt.birth()].data() - f[pt.death()].data() != f[0].data() - f[f.size()-1].data()) {
          int idx = pt.data;
          int l_bound = pt.birth();
          int u_bound = pt.death();
          std::cout << idx << " -> " <<  f.index(f[pt.death()]) << " " << f[idx].data() << " " << f[pt.birth()].data() << " " << f[pt.death()].data() << std::endl;
          Graph::vertex_descriptor u = boost::add_vertex(g);
          g[u].vertex_id = vertexToString(f[pt.birth()][0]);
          buildSubgraph(pt.birth(), f, g, l_bound, u_bound, u);
        };
	      output(i,0) = f[pt.birth()].data();
	      output(i,1) = f[pt.death()].data();
        i++;
      };
      boost::dynamic_properties dp;
      dp.property("weight", boost::get(boost::edge_weight_t(), g));
      boost::write_graphml(outputFile, g, boost::get(&vertex_prop::vertex_id, g), dp);
    };
};

class WassersteinDistanceOp : public OpKernel
{
  public:
    explicit WassersteinDistanceOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
      using PairVector = std::vector<std::pair<double, double>>;
      PairVector diagram_A, diagram_B;
      hera::AuctionParams<float> params;
      params.wasserstein_power = 2.0;
      params.delta = 0.01;
      params.initial_epsilon = 0.0;
      params.epsilon_common_ratio = 0.0;
      params.max_num_phases = 30;
      params.gamma_threshold = 0.0;
      params.max_bids_per_round = 0;

      int num_inputs = context->num_inputs();

      const Tensor& l_input = context->input(num_inputs-3);
      auto layers = l_input.flat<int>();

      const Tensor& a_input = context->input(num_inputs-2);
      auto alphas = a_input.tensor<float, 2>();

      const Tensor& h_dim = context->input(num_inputs-1);
      auto homology_dimension_tensor = h_dim.flat<int>();
      int homology_dimension = homology_dimension_tensor(0);

      Fltr ff;
      std::map<Vertex,float> simplexMapF;

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
      std::map<Vertex,float> simplexMapS;
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
      std::cout << "Second persistence initialized" << std::endl;
      StandardReduction<Persistence> reduceS(persistenceS);
      std::cout << "reducing fs" << std::endl;
      std::cout << "Size of second filration: " << fs.size() << std::endl;
      reduceS(fs);
      std::cout << "Second Persistence initialized" << std::endl;

      auto dgmss = init_diagrams(persistenceS, fs, [&](const Smplx& s) -> float {  return fs.index(s); },
                   [](Persistence::Index i) { return i; });

      std::cout << "Computing Wasserstein Distance via Hera" << std::endl;
      double wasserstein = hera::wasserstein_dist(dgmsf[homology_dimension], dgmss[homology_dimension], params);
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
REGISTER_KERNEL_BUILDER(Name("InputGraphCoPersistence").Device(DEVICE_CPU), InputGraphCoPersistenceOp);
REGISTER_KERNEL_BUILDER(Name("InputGraphFiltration").Device(DEVICE_CPU), InputGraphFiltrationOp);
REGISTER_KERNEL_BUILDER(Name("PersistentSubGraph").Device(DEVICE_CPU), PersistentSubGraphOp);
REGISTER_KERNEL_BUILDER(Name("WassersteinDistance").Device(DEVICE_CPU), WassersteinDistanceOp);
REGISTER_KERNEL_BUILDER(Name("LayerwisePercentile").Device(DEVICE_CPU), LayerwisePercentileOp);
