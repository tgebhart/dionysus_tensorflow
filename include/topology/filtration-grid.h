#ifndef __FILTRATION_GRID_H__
#define __FILTRATION_GRID_H__

#include <vector>
#include <string>
#include "simplex.h"
#include "filtration.h"
#include <boost/iterator/counting_iterator.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>

class FiltrationGrid
{
    public:

      typedef         unsigned int      GridCoord;
      typedef         unsigned int      GridIndex;
      typedef std::pair<bool, RealType> FiltrationValue;
      typedef std::vector<RealType> VertexValues;
      typedef std::vector<GridCoord> SizeVector;
      typedef std::vector<GridCoord> VertexCoordinates;

      typedef Simplex<GridIndex, RealType> TriangulatedSimplex;
      typedef Filtration<TriangulatedSimplex> TriangulatedFiltration;

      FiltrationGrid() {};

      template<class Iterator>
      FiltrationGrid(Iterator bg, Iterator end);

      GridIndex dimensions() const {return sizes_.size();}
      GridIndex vertex_count() const {return values_.size();}

      GridIndex get_vertex_index(VertexCoordinates coords);

      bool is_valid_coord(VertexCoordinates coords);

      void set_vertex_filtration(GridIndex index, RealType value);
      void set_vertex_filtration(VertexCoordinates coords, RealType value);

      TriangulatedFiltration generate_triangulated_filtration();

    private:

      SizeVector sizes_;
      SizeVector stride_lengths_;
      VertexValues values_;

};



#include "filtration-grid.hpp"

#endif // __FILTRATION_GRID_H__
