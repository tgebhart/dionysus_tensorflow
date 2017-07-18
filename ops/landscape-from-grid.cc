#define BOOST_EXCEPTION_DISABLE
#define BOOST_NO_EXCEPTIONS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "include/topology/filtration-grid.h"
#include "include/topology/persistence-landscape.h"
#include "include/topology/static-persistence.h"

using namespace tensorflow;

REGISTER_OP("LandscapeFromGrid")
    .Input("grid_values: float32")
    .Input("dimension: int32")
    .Input("lambda_depth: int32")
    .Input("sample_points: float32")
    .Output("samples: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      auto dim = c->UnknownDim();
      c->MakeDimForScalarInput(2, &dim);
      c->set_output(0, c->Matrix(c->Dim(c->input(3), 0), dim));
      return Status::OK();
    });

REGISTER_OP("LandscapeFromGridGradient")
    .Input("grid_values: float32")
    .Input("dimension: int32")
    .Input("lambda_depth: int32")
    .Input("sample_points: float32")
    .Input("gradient: float32")
    .Output("output_grad: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

Tensor calculate_sampling(FiltrationGrid::TriangulatedFiltration filt,
                                        unsigned int dimension, unsigned int depth,
                                        const Tensor& t_sample_points)
{
  typedef FiltrationGrid::TriangulatedSimplex Simplex;
  typedef FiltrationGrid::TriangulatedFiltration Filtration;

  auto sample_points = t_sample_points.flat<float>();
  int sample_count = t_sample_points.NumElements();

  TensorShape output_shape;
  output_shape.AddDim(depth);
  output_shape.AddDim(sample_count);

  Tensor output_tensor(DT_FLOAT, output_shape);
  auto output = output_tensor.tensor<float, 2>();

  filt.sort(DataDimensionComparison<Simplex>());
  StaticPersistence<> pers(filt);
  pers.pair_simplices();
  StaticPersistence<>::SimplexMap<Filtration> m = pers.make_simplex_map(filt);

  std::map<unsigned int, PersistenceDiagram<> > dgms;
  init_diagrams(dgms, pers.begin(), pers.end(),
      evaluate_through_map(m, Simplex::DataEvaluator()),
      evaluate_through_map(m, Simplex::DimensionExtractor()));

  PersistenceLandscape ls(dgms[dimension]);
  PersistenceLandscape::const_iterator lambda = ls.begin();

  for (int i = 0; i < depth; i++)
  {
    for (int j = 0; j < sample_count; j++)
    {
      if (lambda != ls.end())
        output(i, j) = lambda->calculate_value(sample_points(j));
      else
        output(i, j) = 0;
    }

    if (lambda != ls.end())
      lambda++;
  }

  return output_tensor;
}


class LandscapeFromGridOp : public OpKernel
{
  public:
    explicit LandscapeFromGridOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
      typedef FiltrationGrid::TriangulatedFiltration Filtration;

      const Tensor& t_grid_values = context->input(0);
      const Tensor& t_dimension = context->input(1);
      const Tensor& t_lambda_depth = context->input(2);
      const Tensor& t_sample_points = context->input(3);

      const TensorShape& grid_shape = t_grid_values.shape();
      std::vector<unsigned int> sizes;

      for (int i = 0; i < grid_shape.dims(); i++)
        sizes.push_back(grid_shape.dim_size(i));

      FiltrationGrid grid(sizes.begin(), sizes.end());

      auto values = t_grid_values.flat<float>();
      auto dimension = t_dimension.scalar<int>();
      auto depth = t_lambda_depth.scalar<int>();

      const float* v_data = values.data();
      for (int i = 0; i < grid.vertex_count(); i++)
        grid.set_vertex_filtration(i, v_data[i]);

      Filtration filt = grid.generate_triangulated_filtration();

      Tensor calculated = calculate_sampling(filt, dimension(0), depth(0), t_sample_points);

      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, calculated.shape(),
                                                     &output_tensor));

      output_tensor->CopyFrom(calculated, calculated.shape());
    }
};

class LandscapeFromGridGradientOp : public OpKernel
{
  public:
    explicit LandscapeFromGridGradientOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
      const Tensor& t_grid_values = context->input(0);
      const Tensor& t_dimension = context->input(1);
      const Tensor& t_lambda_depth = context->input(2);
      const Tensor& t_sample_points = context->input(3);
      const Tensor& t_gradient = context->input(4);

      const TensorShape& grid_shape = t_grid_values.shape();
      std::vector<unsigned int> sizes;

      for (int i = 0; i < grid_shape.dims(); i++)
        sizes.push_back(grid_shape.dim_size(i));

      auto values = t_grid_values.flat<float>();
      auto dimension = t_dimension.scalar<int>();
      auto depth = t_lambda_depth.scalar<int>();
      auto gradient = t_gradient.flat<float>();

      FiltrationGrid grid(sizes.begin(), sizes.end());

      const float* v_data = values.data();
      std::set<float> sorted_data;
      float max_boundary = std::numeric_limits<float>::infinity();

      for (int i = 0; i < grid.vertex_count(); i++)
      {
        grid.set_vertex_filtration(i, v_data[i]);
        sorted_data.insert(v_data[i]);
      }

      float last = 0;
      for (auto it = sorted_data.begin(); it != sorted_data.end(); it++)
      {
        float current = *it - last;
        if (current != 0 && current < max_boundary)
          max_boundary = current;

        last = *it;
      }


      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, t_grid_values.shape(),
                                                     &output_tensor));

      auto output = output_tensor->flat<float>();

      for (int i = 0; i < grid.vertex_count(); i++)
      {
        std::vector<float> vec_a;
        std::vector<float> vec_b;
        grid.set_vertex_filtration(i, v_data[i]-max_boundary);
        auto a = calculate_sampling(grid.generate_triangulated_filtration(), dimension(0), depth(0), t_sample_points).flat<float>();

        for (int j = 0; j < t_gradient.NumElements(); j++)
          vec_a.push_back(a(j));

        grid.set_vertex_filtration(i, v_data[i]+max_boundary);
        auto b = calculate_sampling(grid.generate_triangulated_filtration(), dimension(0), depth(0), t_sample_points).flat<float>();

        for (int j = 0; j < t_gradient.NumElements(); j++)
          vec_b.push_back(b(j));

        float sum = 0;
        for (int j = 0; j < t_gradient.NumElements(); j++)
          sum += ((vec_b[j] - vec_a[j])*gradient(j))/(2 * max_boundary);

        output(i) = sum;
        grid.set_vertex_filtration(i, v_data[i]);
      }
    }
};

REGISTER_KERNEL_BUILDER(Name("LandscapeFromGrid").Device(DEVICE_CPU), LandscapeFromGridOp);
REGISTER_KERNEL_BUILDER(Name("LandscapeFromGridGradient").Device(DEVICE_CPU), LandscapeFromGridGradientOp);
