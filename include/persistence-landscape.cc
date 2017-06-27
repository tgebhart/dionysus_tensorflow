#include <algorithm>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/nvp.hpp>

#include "utilities/munkres/munkres.h"

using boost::serialization::make_nvp;

template<class D>
bool compBirth(const PDPoint<D>& a, const PDPoint<D>& b) {return a.x() < b.x();}

template<class D>
bool compDeath(const PDPoint<D>& a, const PDPoint<D>& b) {return a.y() > b.y();}

//Sort by increasing birth and decreasing death
template<class D>
bool compBoth(const PDPoint<D>& a, const PDPoint<D>& b)
{
  if (a.x() == b.x())
    return a.y() > b.y();
  else
    return a.x() < b.x();
}

LambdaCriticals::
LambdaCriticals(const LambdaCriticals& other)
{
  points_ = CriticalVector(other.points_);
}

PersistenceLandscape::
PersistenceLandscape(const PersistenceLandscape& other)
{
    lambdas_.reserve(other.size());
    for (PersistenceLandscape::LambdaVector::const_iterator cur = lambdas_.begin();
      cur != lambdas_.end(); ++cur){
        lambdas_.push_back(LambdaCriticals(*cur));
      }
}

template<class D>
PersistenceLandscape::
PersistenceLandscape(const PersistenceDiagram<D>& diagram)
{
  init(diagram);
}

template<class Iterator, class Evaluator>
PersistenceLandscape::
PersistenceLandscape(Iterator bg, Iterator end, const Evaluator& eval)
{
    init(bg, end, eval);
}

template<class Iterator, class Evaluator>
void
PersistenceLandscape::
init(Iterator bg, Iterator end, const Evaluator& evaluator)
{
    init(PeristenceDiagram(bg, end, evaluator));
}

template<class D>
void
PersistenceLandscape::
init(const PersistenceDiagram<D>& diagram)
{
  typedef PDPoint<D> Point;
  typedef const PDPoint<D> const_Point;
  typedef const CriticalPoint const_CriticalPoint;
  RealType infty = std::numeric_limits<double>::infinity();

  std::list<Point> birth_deaths(diagram.begin(), diagram.end());
  birth_deaths.sort(compBoth<D>);

  int k = 1;
  while (!birth_deaths.empty())
  {
    LambdaCriticals current_crits(k);

    Point front = birth_deaths.front();
    birth_deaths.pop_front();
    typename std::list<Point>::iterator p = birth_deaths.begin();

    if (( (const_Point) front).x() == -infty && ( (const_Point) front).y() == infty)
    {
      current_crits.push_back(CriticalPoint(-infty, infty));
      current_crits.push_back(CriticalPoint(infty, infty));
    }
    else
    {
      if (((const_Point) front).x() == -infty)
        current_crits.push_back(CriticalPoint(-infty, infty));
      else
      {
        current_crits.push_back(CriticalPoint(-infty, 0));
        current_crits.push_back(CriticalPoint(( (const_Point) front).x(), 0));
        current_crits.push_back(CriticalPoint((( (const_Point) front).x() + ( (const_Point) front).y())/2,
          (( (const_Point) front).y() - ( (const_Point) front).x())/2));
      }
    }


    while (!(( (const_CriticalPoint) (current_crits.back())).x() == infty && ( (const_CriticalPoint) (current_crits.back())).y() == 0) &&
          !(( (const_CriticalPoint) (current_crits.back())).x() == infty && ( (const_CriticalPoint) (current_crits.back())).y() == infty))
    {
      typename std::list<Point>::iterator q = std::upper_bound(p, birth_deaths.end(), front, compDeath<D>);
      if (q == birth_deaths.end())
      {
        current_crits.push_back(CriticalPoint(((const_Point)front).y(), 0));
        current_crits.push_back(CriticalPoint(infty, 0));
      }
      else
      {
        Point next(*q);
        p = birth_deaths.erase(q);

        if (( (const_Point) next).x() > ( (const_Point) front).y())
          current_crits.push_back(CriticalPoint(( (const_Point) front).y(), 0));

        if(( (const_Point) next).x() >= ( (const_Point)front).y())
          current_crits.push_back(CriticalPoint(( (const_Point) next).x(), 0));
        else
        {
          current_crits.push_back(CriticalPoint((( (const_Point) next).x() + ((const_Point) front).y())/2,
            (( (const_Point) front).y() - ( (const_Point) next).x())/2));

          Point corner(( (const_Point) next).x(), ( (const_Point) front).y());
          typename std::list<Point>::iterator r = std::upper_bound(p, birth_deaths.end(), corner, compBoth<D>);
          birth_deaths.insert(r, corner);
          p = r;
        }

        if(( (const_Point) next).y() == infty)
          current_crits.push_back(CriticalPoint(infty, infty));
        else
        {
          current_crits.push_back(CriticalPoint((( (const_Point) next).x() + ( (const_Point) next).y())/2,
            (( (const_Point) next).y() - ( (const_Point) next).x())/2));
          front = next;
        }
      }
    }

    this->push_back(current_crits);
    ++k;
  }
}

std::ostream&
LambdaCriticals::
operator<<(std::ostream& out) const
{
    for (const_iterator cur = begin(); cur != end(); ++cur)
        out << *cur << std::endl;
    return out;
}

std::ostream&
PersistenceLandscape::
operator<<(std::ostream& out) const
{
    for (const_iterator cur = begin(); cur != end(); ++cur)
        out << *cur << std::endl;
    return out;
}

template<class Archive>
void
CriticalPoint::
serialize(Archive& ar, version_type )
{
    ar & make_nvp("x", x());
    ar & make_nvp("y", y());
}

template<class Archive>
void
LambdaCriticals::
serialize(Archive& ar, version_type )
{
    ar & make_nvp("points", points_);
}

template<class Archive>
void
PersistenceLandscape::
serialize(Archive& ar, version_type )
{
    ar & make_nvp("lambdas", lambdas_);
}
