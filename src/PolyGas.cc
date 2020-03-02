/*
 * PolyGas.cc
 *
 *  Created on: Mar 26, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "PolyGas.hh"

#include "legion.h"

#include "Vec2.hh"
#include "MyLegion.hh"
#include "InputFile.hh"
#include "Hydro.hh"
#include "Mesh.hh"

using namespace std;
using namespace Legion;


namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
    {
      TaskVariantRegistrar registrar(TID_CALCSTATEHALF, "CPU calcstatehalf");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<PolyGas::calcStateHalfTask>(registrar, "calcstatehalf");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCSTATEHALF, "OMP calcstatehalf");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<PolyGas::calcStateHalfOMPTask>(registrar, "calcstatehalf");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCFORCEPGAS, "CPU calcforcepgas");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<PolyGas::calcForceTask>(registrar, "calcforcepgas");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCFORCEPGAS, "CPU calcforcepgas");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<PolyGas::calcForceOMPTask>(registrar, "calcforcepgas");
    }
}
}; // namespace


PolyGas::PolyGas(const InputFile* inp, Hydro* h) : hydro(h) {
    gamma = inp->getDouble("gamma", 5. / 3.);
    ssmin = inp->getDouble("ssmin", 0.);
}


void PolyGas::calcStateHalfTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double gamma = args[0];
    const double ssmin = args[1];
    const double dt    = task->futures[0].get_result<double>();

    const AccessorRO<double> acc_zr(regions[0], FID_ZR);
    const AccessorRO<double> acc_zvolp(regions[0], FID_ZVOLP);
    const AccessorRO<double> acc_zvol0(regions[0], FID_ZVOL0);
    const AccessorRO<double> acc_ze(regions[0], FID_ZE);
    const AccessorRO<double> acc_zwrate(regions[0], FID_ZWRATE);
    const AccessorRO<double> acc_zm(regions[0], FID_ZM);
    const AccessorWD<double> acc_zp(regions[1], FID_ZP);
    const AccessorWD<double> acc_zss(regions[1], FID_ZSS);

    const double dth = 0.5 * dt;
    const double gm1 = gamma - 1.;
    const double ssmin2 = max(ssmin * ssmin, 1.e-99);
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    for (PointIterator itz(runtime, isz); itz(); itz++)
    {
        // compute EOS at beginning of time step
        const double r = acc_zr[*itz];
        const double e = max(acc_ze[*itz], 0.);
        const double p = gm1 * r * e;
        const double pre = gm1 * e;
        const double per = gm1 * r;
        const double csqd = max(ssmin2, pre + per * p / (r * r));
        const double ss = sqrt(csqd);

        // now advance pressure to the half-step
        const double minv = 1. / acc_zm[*itz];
        const double volp = acc_zvolp[*itz];
        const double vol0 = acc_zvol0[*itz];
        const double wrate = acc_zwrate[*itz];
        const double dv = (volp - vol0) * minv;
        const double bulk = r * csqd;
        const double denom = 1. + 0.5 * per * dv;
        const double src = wrate * dth * minv;
        acc_zp[*itz] = p + (per * src - r * bulk * dv) / denom;
        acc_zss[*itz] = ss;
    }
}


void PolyGas::calcStateHalfOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double gamma = args[0];
    const double ssmin = args[1];
    const double dt    = task->futures[0].get_result<double>();

    const AccessorRO<double> acc_zr(regions[0], FID_ZR);
    const AccessorRO<double> acc_zvolp(regions[0], FID_ZVOLP);
    const AccessorRO<double> acc_zvol0(regions[0], FID_ZVOL0);
    const AccessorRO<double> acc_ze(regions[0], FID_ZE);
    const AccessorRO<double> acc_zwrate(regions[0], FID_ZWRATE);
    const AccessorRO<double> acc_zm(regions[0], FID_ZM);
    const AccessorWD<double> acc_zp(regions[1], FID_ZP);
    const AccessorWD<double> acc_zss(regions[1], FID_ZSS);

    const double dth = 0.5 * dt;
    const double gm1 = gamma - 1.;
    const double ssmin2 = max(ssmin * ssmin, 1.e-99);
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    #pragma omp parallel for schedule(static, OMP_CHUNK_SIZE)
    for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
    {
        // compute EOS at beginning of time step
        const double r = acc_zr[z];
        const double e = max(acc_ze[z], 0.);
        const double p = gm1 * r * e;
        const double pre = gm1 * e;
        const double per = gm1 * r;
        const double csqd = max(ssmin2, pre + per * p / (r * r));
        const double ss = sqrt(csqd);

        // now advance pressure to the half-step
        const double minv = 1. / acc_zm[z];
        const double volp = acc_zvolp[z];
        const double vol0 = acc_zvol0[z];
        const double wrate = acc_zwrate[z];
        const double dv = (volp - vol0) * minv;
        const double bulk = r * csqd;
        const double denom = 1. + 0.5 * per * dv;
        const double src = wrate * dth * minv;
        acc_zp[z] = p + (per * src - r * bulk * dv) / denom;
        acc_zss[z] = ss;
    }
}


void PolyGas::calcForceTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double2> acc_ssurf(regions[0], FID_SSURFP);
    const AccessorRO<double> acc_zp(regions[1], FID_ZP);
    const AccessorWD<double2> acc_sf(regions[2], FID_SFP);

    const IndexSpace& iss = task->regions[0].region.get_index_space();

    for (PointIterator its(runtime, iss); its(); its++)
    {
        const Pointer z = acc_mapsz[*its];
        const double p = acc_zp[z];
        const double2 surf = acc_ssurf[*its];
        const double2 sfx = -p * surf;
        acc_sf[*its] = sfx;
    }
}


void PolyGas::calcForceOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double2> acc_ssurf(regions[0], FID_SSURFP);
    const AccessorRO<double> acc_zp(regions[1], FID_ZP);
    const AccessorWD<double2> acc_sf(regions[2], FID_SFP);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp parallel for schedule(static, OMP_CHUNK_SIZE)
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        const Pointer z = acc_mapsz[s];
        const double p = acc_zp[z];
        const double2 surf = acc_ssurf[s];
        const double2 sfx = -p * surf;
        acc_sf[s] = sfx;
    }
}

