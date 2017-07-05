#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "include/utilities/log.h"

#include "include/topology/simplex.h"
#include "include/topology/filtration.h"
#include "include/topology/static-persistence.h"
#include "include/topology/dynamic-persistence.h"
#include "include/topology/persistence-diagram.h"
#include "include/topology/persistence-landscape.h"
#include <include/utilities/indirect.h>

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

REGISTER_OP("LandscapeTest")
    .Input("input_tensor: int32")
    .Output("output_tensor: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

typedef unsigned Vertex;
typedef Simplex<Vertex, double> Smplx;
typedef Filtration<Smplx> Fltr;
// typedef         StaticPersistence<> Persistence;
typedef DynamicPersistenceTrails<> Persistence;
typedef PersistenceDiagram<> PDgm;
typedef OffsetBeginMap<Persistence, Fltr, Persistence::iterator, Fltr::Index>
    PersistenceFiltrationMap;
typedef OffsetBeginMap<Fltr, Persistence, Fltr::Index, Persistence::iterator>
    FiltrationPersistenceMap;

// Transposes elements of the filtration together with the
struct FiltrationTranspositionVisitor
    : public Persistence::TranspositionVisitor {
  typedef Persistence::iterator iterator;

  FiltrationTranspositionVisitor(const Persistence& p, Fltr& f)
      : p_(p), f_(f) {}
  void transpose(iterator i) { f_.transpose(f_.begin() + (i - p_.begin())); }

  const Persistence& p_;
  Fltr& f_;
};

void fillTriangleSimplices(Fltr& c) {
  typedef std::vector<Vertex> VertexVector;
  VertexVector vertices(4);
  vertices[0] = 0;
  vertices[1] = 1;
  vertices[2] = 2;
  vertices[3] = 0;

  VertexVector::const_iterator bg = vertices.begin();
  VertexVector::const_iterator end = vertices.end();
  c.push_back(Smplx(bg, bg + 1, 0));        // 0 = A
  c.push_back(Smplx(bg + 1, bg + 2, 1));    // 1 = B
  c.push_back(Smplx(bg + 2, bg + 3, 2));    // 2 = C
  c.push_back(Smplx(bg, bg + 2, 2.5));      // AB
  c.push_back(Smplx(bg + 1, bg + 3, 2.9));  // BC
  c.push_back(Smplx(bg + 2, end, 3.5));     // CA
  c.push_back(Smplx(bg, bg + 3, 5));        // ABC
}

class LandscapeTestOp : public OpKernel {
public:
  explicit LandscapeTestOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // get input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    Fltr f;
    fillTriangleSimplices(f);
    std::cout << "Simplices filled" << std::endl;
    for (Fltr::Index cur = f.begin(); cur != f.end(); ++cur)
      std::cout << "  " << *cur << std::endl;


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

    PersistenceFiltrationMap pfmap(p, f);
    DimensionFunctor<PersistenceFiltrationMap, Fltr> dim(pfmap, f);

    // Transpositions
    FiltrationPersistenceMap fpmap(f, p);
    FiltrationTranspositionVisitor visitor(p, f);
    Smplx A;
    A.add(0);
    std::cout << A << std::endl;
    std::cout << "Transposing A: " << p.transpose(fpmap[f.find(A)], dim, visitor)
              << std::endl;  // 1.2 unpaired

    Smplx BC;
    BC.add(1);
    BC.add(2);
    Smplx AB;
    AB.add(0);
    AB.add(1);
    std::cout << BC << std::endl;
    std::cout << p.transpose(fpmap[f.find(BC)], dim, visitor)
              << std::endl;  // 3.1
    // p.transpose(fpmap[f.find(BC)], dim, visitor);
    std::cout << AB << std::endl;
    std::cout << p.transpose(fpmap[f.find(AB)], dim, visitor)
              << std::endl;  // 2.1
    // p.transpose(fpmap[f.find(AB)], dim, visitor);

    std::cout << p.transpose(p.begin(), dim, visitor)
              << std::endl;  // transposition case 1.2 special
    std::cout << p.transpose(boost::next(p.begin()), dim, visitor) << std::endl;
    std::cout << p.transpose(boost::next(p.begin(), 3), dim, visitor)
              << std::endl;

    std::cout << "Landscape: " << std::endl;

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
}

REGISTER_KERNEL_BUILDER(Name("LandscapeTest").Device(DEVICE_CPU), LandscapeTestOp);
