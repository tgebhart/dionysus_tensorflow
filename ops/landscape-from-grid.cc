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


class LandscapeFromGridOp : public OpKernel
{
  public:
    explicit LandscapeFromGridOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
      typedef FiltrationGrid::TriangulatedSimplex Simplex;
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

      FiltrationGrid::VertexCoordinates coords = grid.first();

      auto values = t_grid_values.flat<float>();
      auto dimension = t_dimension.scalar<int>();
      auto depth = t_lambda_depth.scalar<int>();
      auto sample_points = t_sample_points.flat<float>();
      int sample_count = t_sample_points.NumElements();

      TensorShape output_shape;
      output_shape.AddDim(sample_count);
      output_shape.AddDim(depth(0));
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

      auto output = output_tensor->tensor<float, 2>();


      const float* v_data = values.data();
      for (int i = 0; i < grid.vertex_count(); i++)
        grid.set_vertex_filtration(i, v_data[i]);

      Filtration filt = grid.generate_triangulated_filtration();

      filt.sort(DataDimensionComparison<Simplex>());
      StaticPersistence<> pers(filt);
      pers.pair_simplices();
      StaticPersistence<>::SimplexMap<Filtration> m = pers.make_simplex_map(filt);

      std::map<unsigned int, PersistenceDiagram<> > dgms;
      init_diagrams(dgms, pers.begin(), pers.end(),
                    evaluate_through_map(m, Simplex::DataEvaluator()),
                    evaluate_through_map(m, Simplex::DimensionExtractor()));

      PersistenceLandscape ls(dgms[dimension(0)]);
      PersistenceLandscape::const_iterator lambda = ls.begin();

      for (int i = 0; i < depth(0); i++)
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
    }
};

REGISTER_KERNEL_BUILDER(Name("LandscapeFromGrid").Device(DEVICE_CPU), LandscapeFromGridOp);
