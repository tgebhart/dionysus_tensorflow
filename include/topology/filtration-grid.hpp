#include "filtration-grid.h"
#include <string>
#include <sstream>

template<class Iterator>
FiltrationGrid::
FiltrationGrid(Iterator bg, Iterator end)
{
  for (Iterator cur = bg; cur != end; ++cur)
    sizes_.push_back(*cur);

  unsigned int total_size = 1;

  for (SizeVector::const_iterator cur = sizes_.begin(); cur != sizes_.end(); cur++)
  {
    stride_lengths_.push_back(total_size);
    total_size *= (*cur);
  }

  values_.reserve(total_size);

  for (GridIndex i = 0; i < total_size; i++)
    values_.push_back(-Infinity);
}


//Gets the index of a vertex, when considering coordinates relative to vertices
FiltrationGrid::GridIndex
FiltrationGrid::
get_vertex_index(VertexCoordinates coords)
{
  /* if (coords.size() != dimensions()) throw... */

  GridIndex index = 0;

  for (GridIndex i = 0; i < dimensions(); i++)
    index += stride_lengths_[i] * coords[i];

  return index;
}

bool
FiltrationGrid::
is_valid_coord(VertexCoordinates coords)
{
  /* if (coords.size() != dimensions()) throw... */

  for (GridIndex i = 0; i < dimensions(); i++)
  {
    if (coords[i] >= sizes_[i])
      return false;
  }

  return true;
}

void
FiltrationGrid::
set_vertex_filtration(GridIndex index, RealType value)
{
  values_[index] = value;
}

void
FiltrationGrid::
set_vertex_filtration(VertexCoordinates coords, RealType value)
{
  set_vertex_filtration(get_vertex_index(coords), value);
}


FiltrationGrid::TriangulatedFiltration
FiltrationGrid::
generate_triangulated_filtration()
{
  TriangulatedFiltration filt;

  VertexCoordinates vert_coords;
  std::vector< GridIndex > perm_string;

  for (GridIndex i = 0; i < dimensions(); i++)
  {
    vert_coords.push_back(0);
    perm_string.push_back(i);
  }

  for (GridIndex i = 0; i < values_.size(); i++)
  {
    for (GridIndex j = 0; j < dimensions(); j++)
    {
      if (vert_coords[j] >= sizes_[j] && j < (dimensions() - 1))
      {
        vert_coords[j + 1]++;
        vert_coords[j] = 0;
      }

      do
      {
        VertexCoordinates c_coords = vert_coords;
        GridIndex c_index = get_vertex_index(c_coords);
        std::vector< GridIndex > indices;
        std::vector< RealType > filt_values;

        indices.push_back(c_index);
        filt_values.push_back(values_[c_index]);
        filt.push_back(TriangulatedSimplex(indices.begin(), indices.end(),
            *std::max_element(filt_values.begin(), filt_values.end())));


        for (GridIndex k = 0; k < dimensions(); k++)
        {
          c_coords[perm_string[k]]++;

          // std::cout << "dimensions: " << dimensions() << std::endl;
          // std::cout << "sizes: " << sizes_[0] << " " << sizes_[1] << std::endl;
          //
          // std::cout << "vert_coords: " << vert_coords[0] << ", " << vert_coords[1] << std::endl;
          // std::cout << "c_coords: " <<  c_coords[0] << ", " << c_coords[1] << std::endl;
          // std::cout << "perm_string: " << perm_string[0] << ", " << perm_string[1] << std::endl;

          if (is_valid_coord(c_coords))
          {
            // std::cout << "c_coords valid" << std::endl;

            c_index = get_vertex_index(c_coords);
            indices.push_back(c_index);
            filt_values.push_back(values_[c_index]);
            filt.push_back(TriangulatedSimplex(indices.begin(), indices.end(),
                *std::max_element(filt_values.begin(), filt_values.end())));
          }
          else
            break;
        }
      } while(std::next_permutation(perm_string.begin(), perm_string.end()));
    }

    vert_coords[0]++;
  }

  return filt;
}
