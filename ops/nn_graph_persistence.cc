#define BOOST_EXCEPTION_DISABLE
#define BOOST_NO_EXCEPTIONS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "include/topology/simplex.h"
#include "include/topology/filtration.h"
#include "include/topology/filtration-grid.h"
#include "include/topology/static-persistence.h"
#include "include/topology/persistence-diagram.h"

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

typedef std::tuple<int, int, int, int> Vertex;
typedef Simplex<Vertex, float> Smplx;
typedef Filtration<Smplx> Fltr;
typedef StaticPersistence<> Persistence;
typedef PersistenceDiagram<> PDgm;

REGISTER_OP("InputGraphPersistence")
    .Attr("T: list({float})")
    .Input("in_tensors: T")
    .Input("in_layers: int32")
    .Output("output_tensor: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

std::vector<std::tuple<int, int, int, int>> vertexId(int height, int width, int channel, int layer) {
  std::vector<std::tuple<int, int, int, int>> v = {std::make_tuple(height, width, channel, layer)};
  return v;
};

std::vector<std::tuple<int, int, int, int>> oneVertexId(int h, int w, int c, int l, int i, int x, int d, int m) {
  std::vector<std::tuple<int, int, int, int>> v = {std::make_tuple(h, w, c, l),
                                                  std::make_tuple(i, x, d, m)};
  return v;
};


void addConvLayer(const Tensor& input_tensor, const Tensor& filter_tensor,
                  const Tensor& output_tensor, Fltr& f, int first_layer,
                  int second_layer, int idx) {

  std::cout << input_tensor.DebugString() << std::endl;

  const TensorShape& input_shape = input_tensor.shape();
  const TensorShape& output_shape = output_tensor.shape();
  const TensorShape& filter_shape = filter_tensor.shape();

  auto input = input_tensor.tensor<float,4>();
  auto output = output_tensor.tensor<float,4>();
  auto filter = filter_tensor.tensor<float,4>();

  for (int ih = 0; ih < input_shape.dim_size(1); ++ih) {
    for (int iw = 0; iw < input_shape.dim_size(2); ++iw) {
      for (int ic = 0; ic < input_shape.dim_size(3); ++ic) {
        f.push_back(Smplx(vertexId(ih, iw, ic, first_layer), 0.0f));
        for (int oc = 0; oc < output_shape.dim_size(3); ++oc) {
          int mid_h = filter_shape.dim_size(0)/2;
          int mid_w = filter_shape.dim_size(1)/2;
          f.push_back(Smplx(vertexId(ih, iw, oc, second_layer), 0.0f));
          float phi = filter(mid_h,mid_w,ic,oc) * input(ih,iw,ic,oc);
          f.push_back(Smplx(oneVertexId(ih,iw,ic,first_layer,ih,iw,oc,second_layer), phi));
          for (int h = -mid_h; h < mid_h; ++h) {
            int h_idx = ih + h;
            if (h_idx >= 0 && h_idx < input_shape.dim_size(1)) {
              for (int w = -mid_w; w < mid_w; ++w) {
                int w_idx = iw + w;
                if (w_idx >= 0 && w_idx < input_shape.dim_size(2)) {
                  phi = filter(mid_h+h,mid_w+w,ic,oc) * input(idx,h_idx,w_idx,ic);
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

void addFCLayerFourTwo(const Tensor& input_tensor, const Tensor& weight_tensor,
                const Tensor& output_tensor, Fltr& f, int first_layer,
                int second_layer, int idx) {



  const TensorShape& input_shape = input_tensor.shape();
  const TensorShape& output_shape = output_tensor.shape();
  const TensorShape& weight_shape = weight_tensor.shape();

  std::cout << input_shape.dims() << std::endl;

  auto input = input_tensor.tensor<float,4>();
  auto output = output_tensor.tensor<float,2>();
  auto weight = weight_tensor.tensor<float,2>();

  if (input_shape.dims() <= 2) {
      std::cout << "Wrong FC Layer Shape Method!" << std::endl;
  };

  int channels = input_shape.dim_size(3);
  int hs = input_shape.dim_size(1);
  int ws = input_shape.dim_size(2);

  for (int w = 0; w < weight_shape.dim_size(1); ++w) {
    f.push_back(Smplx(vertexId(0,w,0,second_layer), 0.0f));
    for (int c = 0; c < channels; ++c) {
      for (int fh = 0; fh < hs; ++fh) {
        for (int fw = 0; fw < ws; ++fw) {
          f.push_back(Smplx(vertexId(fh,fw,c,first_layer), 0.0f));
          int channel_offset = ((ws * fh) + fw) + (ws * hs * c);
          float phi = 0.0f;
          phi = input(idx,fh,fw,c) * weight(channel_offset,w);
          f.push_back(Smplx(oneVertexId(fh,fw,c,first_layer,0,w,0,second_layer), phi));
        };
      };
    };
  };
};

void addFCLayerTwoTwo(const Tensor& input_tensor, const Tensor& weight_tensor,
                const Tensor& output_tensor, Fltr& f, int first_layer,
                int second_layer, int idx) {



  const TensorShape& input_shape = input_tensor.shape();
  const TensorShape& output_shape = output_tensor.shape();
  const TensorShape& weight_shape = weight_tensor.shape();

  std::cout << input_shape.dims() << std::endl;

  auto input = input_tensor.tensor<float,2>();
  auto output = output_tensor.tensor<float,2>();
  auto weight = weight_tensor.tensor<float,2>();

  if (input_shape.dims() != 2) {
    std::cout << "Wrong FC Layer Shape Method!" << std::endl;
  };

  int ws = input_shape.dim_size(1);

  for (int w = 0; w < weight_shape.dim_size(1); ++w) {
    f.push_back(Smplx(vertexId(0,w,0,second_layer), 0.0f));
    for (int fw = 0; fw < ws; ++fw) {
      f.push_back(Smplx(vertexId(0,fw,0,first_layer), 0.0f));
      float phi = 0.0f;
      phi = input(idx,fw) * weight(fw,w);
      f.push_back(Smplx(oneVertexId(0,fw,0,first_layer,0,w,0,second_layer), phi));
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

      const Tensor& l_input = context->input(num_inputs-1);
      auto layers = l_input.flat<int>();

      int numLayers = layers.dimension(0);

      Fltr f;

      int layerType;
      int layer = 0;
      for (int i = 1; i < numLayers; ++i) {
        if (i % (3-1) == 0) {
          layerType = layers(i);
          if (layerType == 2) {
            addConvLayer(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0);
          };
          if (layerType == 4) {
            if (layers(i-2) == 2 || layers(i-2) == 0) {
                addFCLayerFourTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0);
            } else {
                addFCLayerTwoTwo(context->input(i-2), context->input(i-1), context->input(i), f, layer, layer+1, 0);
            };
          };
          layer++;
        };
      };

      f.sort(Smplx::DataComparison());
      std::cout << "Filtration initialized" << std::endl;
      // // std::cout << f << std::endl;

      Persistence p(f);
      std::cout << "Persistence initialized" << std::endl;

      p.pair_simplices();
      std::cout << "Simplices paired" << std::endl;

      Persistence::SimplexMap<Fltr> m = p.make_simplex_map(f);
      std::map<Dimension, PDgm> dgms;
      init_diagrams(dgms, p.begin(), p.end(),
                    evaluate_through_map(m, Smplx::DataEvaluator()),
                    evaluate_through_map(m, Smplx::DimensionExtractor()));
      //

      for(auto elem : dgms) {
        std::cout << elem.first << " " << typeid(elem.second).name() << "\n";
      }

      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, l_input.shape(),
                                                     &output_tensor));

    }
};



REGISTER_KERNEL_BUILDER(Name("InputGraphPersistence").Device(DEVICE_CPU), InputGraphPersistenceOp);
