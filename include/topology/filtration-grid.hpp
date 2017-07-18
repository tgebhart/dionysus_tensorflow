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
get_vertex_index(const VertexCoordinates& coords) const
{
  /* if (coords.size() != dimensions()) throw... */

  GridIndex index = 0;

  for (GridIndex i = 0; i < dimensions(); i++)
    index += stride_lengths_[i] * coords[i];

  return index;
}

FiltrationGrid::VertexCoordinates
FiltrationGrid::
get_vertex_coordinates(const GridIndex& ind) const
{
  VertexCoordinates coords;
  GridIndex j = ind;

  for (GridIndex i = 0; i < dimensions(); i++)
  {
    coords.push_back(j % sizes_[i]);
    j = (GridIndex)std::floor(j / sizes_[i]);
  }

  return coords;
}


FiltrationGrid::VertexCoordinates
FiltrationGrid::
increment_coordinates(VertexCoordinates coords) const
{
  coords[0]++;
  for (GridIndex i = 0; i < dimensions(); i++)
  {
    if (coords[i] >= sizes_[i])
    {
      if (i == (dimensions() - 1))
      {
        return out_of_grid();
      }
      else
      {
        coords[i + 1]++;
        coords[i] = 0;
      }
    }
  }

  return coords;
}

FiltrationGrid::VertexCoordinates
FiltrationGrid::
decrement_coordinates(VertexCoordinates coords) const
{
  coords[0]--;
  for (GridIndex i = 0; i < dimensions(); i++)
  {
    if (coords[i] <= 0)
    {
      if (i == (dimensions() - 1))
      {
        return first();
      }
      else
      {
        coords[i + 1]--;
        coords[i] = sizes_[i] - 1;
      }
    }
  }

  return coords;
}

FiltrationGrid::VertexCoordinates
FiltrationGrid::
first() const
{
  VertexCoordinates coords;

  for (GridIndex i = 0; i < dimensions(); i++)
    coords.push_back(0);

  return coords;
}

FiltrationGrid::VertexCoordinates
FiltrationGrid::
last() const
{
  VertexCoordinates coords;

  for (GridIndex i = 0; i < dimensions(); i++)
    coords.push_back(sizes_[i] - 1);

  return coords;
}

FiltrationGrid::VertexCoordinates
FiltrationGrid::
out_of_grid() const
{
  VertexCoordinates coords;

  for (GridIndex i = 0; i < dimensions(); i++)
    coords.push_back(std::numeric_limits<GridIndex>::max());

  return coords;
}

bool
FiltrationGrid::
is_valid_coord(const VertexCoordinates& coords) const
{
  /* if (coords.size() != dimensions()) throw... */

  for (GridIndex i = 0; i < dimensions(); i++)
  {
    if (coords[i] < 0 || coords[i] >= sizes_[i])
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

  do
  {
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

        if (is_valid_coord(c_coords))
        {

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

    vert_coords = increment_coordinates(vert_coords);

  }while (vert_coords != out_of_grid());

  return filt;
}

std::ostream&
FiltrationGrid::
operator<<(std::ostream& out) const
{
    std::copy(begin(values_), end(values_), std::ostream_iterator<RealType>(out, " "));
    return out;
}
