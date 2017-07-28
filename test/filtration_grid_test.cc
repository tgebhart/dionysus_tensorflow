#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "include/utilities/log.h"
#include "include/utilities/indirect.h"

#include "include/topology/simplex.h"
#include "include/topology/filtration.h"
#include "include/topology/filtration-grid.h"
#include "include/topology/static-persistence.h"
#include "include/topology/dynamic-persistence.h"
#include "include/topology/persistence-diagram.h"
#include "include/topology/persistence-landscape.h"

#include <vector>
#include <map>
#include <iostream>

#if 1
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#endif

using namespace tensorflow;

REGISTER_OP("FiltrationGridTest")
    .Input("input_tensor: int32")
    .Output("output_tensor: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class FiltrationGridTestOp : public OpKernel {
public:
  explicit FiltrationGridTestOp(OpKernelConstruction* context) : OpKernel(context) {}

  typedef unsigned int Vertex;
  typedef Simplex<Vertex, RealType> Smplx;
  typedef Filtration<Smplx> Fltr;
  typedef StaticPersistence<> Persistence;
  // typedef         DynamicPersistenceTrails<> Persistence;
  typedef PersistenceDiagram<> PDgm;
  typedef OffsetBeginMap<Persistence, Fltr, Persistence::iterator, Fltr::Index>
      PersistenceFiltrationMap;
  typedef OffsetBeginMap<Fltr, Persistence, Fltr::Index, Persistence::iterator>
      FiltrationPersistenceMap;

  void fillGrid(FiltrationGrid& g) {
    for (FiltrationGrid::GridIndex i = 0; i < g.vertex_count(); i++)
      g.set_vertex_filtration(i, rand() % 200);
  }

  void Compute(OpKernelContext* context) override {
    // get input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    FiltrationGrid::SizeVector sizes;
    sizes.push_back(12);
    sizes.push_back(12);

    FiltrationGrid g(sizes.begin(), sizes.end());

    fillGrid(g);

    std::cout << g.dimensions() << std::endl;

    Fltr f = g.generate_triangulated_filtration();

    std::cout << "Simplices filled" << std::endl;
    for (Fltr::Index cur = f.begin(); cur != f.end(); ++cur)
      std::cout << "  " << *cur << std::endl;

  // #if 1  // testing serialization of the Filtration (really Simplex)
  //   {
  //     std::ofstream ofs("complex");
  //     boost::archive::text_oarchive oa(ofs);
  //     oa << f;
  //     f.clear();
  //   }
  //
  //   {
  //     std::ifstream ifs("complex");
  //     boost::archive::text_iarchive ia(ifs);
  //     ia >> f;
  //   }
  // #endif

    f.sort(Smplx::DataComparison());
    std::cout << "Filtration initialized" << std::endl;
    std::cout << f << std::endl;

    Persistence p(f);
    std::cout << "Persistence initialized" << std::endl;

    p.pair_simplices();
    std::cout << "Simplices paired" << std::endl;

    Persistence::SimplexMap<Fltr> m = p.make_simplex_map(f);
    std::map<Dimension, PDgm> dgms;
    init_diagrams(dgms, p.begin(), p.end(),
                  evaluate_through_map(m, Smplx::DataEvaluator()),
                  evaluate_through_map(m, Smplx::DimensionExtractor()));

    std::cout << 0 << std::endl << dgms[0] << std::endl;
    std::cout << 1 << std::endl << dgms[1] << std::endl;

    for (std::map<Dimension, PDgm>::iterator it = dgms.begin(); it != dgms.end();
         it++) {
      PersistenceLandscape test(it->second);
      for (PersistenceLandscape::const_iterator i = test.begin(); i != test.end();
           i++) {
        std::cout << "Landscape Start" << std::endl;
        for (LambdaCriticals::const_iterator j = i->begin(); j != i->end(); j++)
          std::cout << *j << std::endl;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("FiltrationGridTest").Device(DEVICE_CPU), FiltrationGridTestOp);
