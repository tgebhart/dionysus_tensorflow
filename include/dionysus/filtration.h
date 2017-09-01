#ifndef DIONYSUS_FILTRATION_H
#define DIONYSUS_FILTRATION_H

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/random_access_index.hpp>

namespace b   = boost;
namespace bmi = boost::multi_index;

namespace dionysus
{

// Filtration stores a filtered cell complex as boost::multi_index_container<...>.
// It allows for bidirectional translation between a cell and its index.
template<class Cell_,
         class CellLookupIndex_ = bmi::hashed_unique<bmi::identity<Cell_>>>
class Filtration
{
    public:
        struct order {};

        typedef             Cell_                                               Cell;
        typedef             CellLookupIndex_                                    CellLookupIndex;

        typedef             b::multi_index_container<Cell, bmi::indexed_by<CellLookupIndex,
                                                                           bmi::random_access<bmi::tag<order>>
                                                                          >>    Container;
        typedef             typename Container::value_type                      value_type;

        typedef             typename Container::template nth_index<0>::type     Complex;
        typedef             typename Container::template nth_index<1>::type     Order;
        typedef             typename Order::const_iterator                      OrderConstIterator;
        typedef             typename Order::iterator                            OrderIterator;


    public:
                            Filtration()                                        = default;
                            Filtration(Filtration&& other)                      = default;
        Filtration&         operator=(Filtration&& other)                       = default;

                            Filtration(const std::initializer_list<Cell>& cells):
                                Filtration(std::begin(cells), std::end(cells))  {}

        template<class Iterator>
                            Filtration(Iterator bg, Iterator end):
                                cells_(bg, end)                                 {}

        template<class CellRange>
                            Filtration(const CellRange& cells):
                                Filtration(std::begin(cells), std::end(cells))  {}

        // Lookup
        const Cell&         operator[](size_t i) const                          { return cells_.template get<order>()[i]; }
        OrderConstIterator  iterator(const Cell& s) const                       { return bmi::project<order>(cells_, cells_.find(s)); }
        size_t              index(const Cell& s) const                          { return iterator(s) - begin(); }
        bool                contains(const Cell& s) const                       { return cells_.find(s) != cells_.end(); }

        void                push_back(const Cell& s)                            { cells_.template get<order>().push_back(s); }
        void                push_back(Cell&& s)                                 { cells_.template get<order>().push_back(s); }

        void                replace(size_t i, const Cell& s)                    { cells_.template get<order>().replace(begin() + i, s); }

        // return index of the cell, adding it, if necessary
        size_t              add(const Cell& s)                                  { size_t i = index(s); if (i == size()) emplace_back(s); return i; }
        size_t              add(Cell&& s)                                       { size_t i = index(s); if (i == size()) emplace_back(std::move(s)); return i; }

        template<class... Args>
        void                emplace_back(Args&&... args)                        { cells_.template get<order>().emplace_back(std::forward<Args>(args)...); }

        template<class Cmp = std::less<Cell>>
        void                sort(const Cmp& cmp = Cmp())                        { cells_.template get<order>().sort(cmp); }

        OrderConstIterator  begin() const                                       { return cells_.template get<order>().begin(); }
        OrderConstIterator  end() const                                         { return cells_.template get<order>().end(); }
        OrderIterator       begin()                                             { return cells_.template get<order>().begin(); }
        OrderIterator       end()                                               { return cells_.template get<order>().end(); }
        size_t              size() const                                        { return cells_.size(); }
        void                clear()                                             { return Container().swap(cells_); }

        Cell&               back()                                              { return const_cast<Cell&>(cells_.template get<order>().back()); }
        const Cell&         back() const                                        { return cells_.template get<order>().back(); }

    private:
        Container           cells_;
};

}

#endif
