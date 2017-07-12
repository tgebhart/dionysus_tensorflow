#ifndef __PERSISTENCE_LANDSCAPE_H__
#define __PERSISTENCE_LANDSCAPE_H__

#include "utilities/types.h"


#include <vector>
#include <iostream>
#include <cmath>

#include <boost/compressed_pair.hpp>
#include <boost/optional.hpp>
#include <boost/serialization/access.hpp>

#include "persistence-diagram.h"


/**
 * Class: CriticalPoint
 *
 * Stores a critical point of a persistence landscape
 */
class CriticalPoint
{
    public:
                                CriticalPoint(const CriticalPoint& other):
                                    point_(other.point_)                    {}

                                CriticalPoint(const RealType x = 0, const RealType y = 0)
                                                                            {point_.first = x; point_.second = y;}

        RealType                x() const                                   { return point_.first; }
        RealType                y() const                                   { return point_.second; }

        std::ostream&           operator<<(std::ostream& out) const         { return (out << x() << " " << y()); } // << " " << data()); }

        bool                    operator<(CriticalPoint b) const            { return x() < b.x(); }

    private:
        RealType&               x()                                         { return point_.first; }
        RealType&               y()                                         { return point_.second; }

    private:
        std::pair<RealType, RealType>       point_;

    private:
        /* Serialization */
        friend class boost::serialization::access;

        template<class Archive>
        void                    serialize(Archive& ar, version_type );
};

std::ostream&                   operator<<(std::ostream& out, const CriticalPoint& point)
{ return (point.operator<<(out)); }

/**
 * Class: LambdaCriticals
 *
 * Stores the critical points of the kth lambda function of a persistence landscape.
 */
class LambdaCriticals
{
    public:
        typedef                 CriticalPoint                                   Point;
        typedef                 std::vector<Point>                              CriticalVector;
        typedef                 typename CriticalVector::const_iterator         const_iterator;

                                LambdaCriticals()                        {}

                                LambdaCriticals( const IndexType& index ):
                                    index_( index )                  {}

                                LambdaCriticals(const LambdaCriticals& other);

        template<class Iterator, class Evaluator>
                                LambdaCriticals(Iterator bg, Iterator end,
                                                   const Evaluator& eval = Evaluator());


        const_iterator          begin() const                                   { return points_.begin(); }
        const_iterator          end() const                                     { return points_.end(); }
        size_t                  size() const                                    { return points_.size(); }

        CriticalPoint           front() const                                   {return points_.front();}
        CriticalPoint           back() const                                    {return points_.back();}

        void                    push_back(const Point& point)                   { points_.push_back(point); }

        std::ostream&           operator<<(std::ostream& out) const;

        IndexType               index() const                                   { return index_; }

        RealType                calculate_value(RealType x) const;

    private:
        CriticalVector          points_;
        IndexType               index_;

    private:
        /* Serialization */
        friend class boost::serialization::access;

        template<class Archive>
        void                    serialize(Archive& ar, version_type );
};

std::ostream&                   operator<<(std::ostream& out, const LambdaCriticals& c)
{ return (c.operator<<(out)); }



/**
 * Class: PersistenceLandscape
 *
 * Stores an ordered list of vectors of critical points of individual lambda functions
 */
class PersistenceLandscape
{
    public:
        typedef                 std::vector<LambdaCriticals>                    LambdaVector;
        typedef                 typename LambdaVector::const_iterator           const_iterator;

                                PersistenceLandscape()                        {}

                                PersistenceLandscape( const Dimension& dimension ):
                                    dimension_( dimension )                  {}

                                PersistenceLandscape(const PersistenceLandscape& other);

        template<class D>
                                PersistenceLandscape(const PersistenceDiagram<D>& diagram);

        template<class Iterator, class Evaluator>
                                PersistenceLandscape(Iterator bg, Iterator end,
                                const Evaluator& eval = Evaluator());

        template<class Iterator, class Evaluator>
        void                    init(Iterator bg, Iterator end,
                                const Evaluator& eval = Evaluator());

        template<class D>
        void                    init(const PersistenceDiagram<D>& diagram);


        const_iterator          begin() const                               { return lambdas_.begin(); }
        const_iterator          end() const                                 { return lambdas_.end(); }
        size_t                  size() const                                { return lambdas_.size(); }

        void                    push_back(const LambdaCriticals& lambda)               { lambdas_.push_back(lambda); }

        std::ostream&           operator<<(std::ostream& out) const;

        Dimension               dimension() const                           { return dimension_; }

    private:
        LambdaVector            lambdas_;
        Dimension               dimension_;

    private:
        /* Serialization */
        friend class boost::serialization::access;

        template<class Archive>
        void                    serialize(Archive& ar, version_type );
};

std::ostream&                   operator<<(std::ostream& out, const PersistenceLandscape& pd)
{ return (pd.operator<<(out)); }


#include "persistence-landscape.hpp"

#endif // __PERSISTENCE_DIAGRAM_H__
