#pragma once

#include <memory>
#include <string>
#include <vector>

#include "setup.h"

class Evolution {
public:
    explicit Evolution(const Setup &setup);

    // Returns the saved field snapshots: phicomplete[i_s][i_z]
    const std::vector<std::vector<double>> &GetPhi()   const { return phicomplete; }
    const std::vector<double>              &GetSlist()  const { return slist; }
    double                                  GetDS()     const { return ds_out; }

private:
    void Evolve();
    void EvolvepiFirstHalfStep(int n_baby);
    double EvolvePi(int i_z, double s, double step) const;

    int    n_z, how_often_ds, baby_steps, n_steps;
    double ds, smax, dz, d;

    const Potential *potential;   // non-owning; Setup owns it

    std::vector<double> z;
    std::vector<double> phi;
    std::vector<double> pi;
    std::vector<double> slist;    // s values for saved snapshots

    // snapshots: phicomplete[i_snapshot][i_z]
    std::vector<std::vector<double>> phicomplete;

    double ds_out; // effective ds between saved snapshots = ds * how_often_ds
};
