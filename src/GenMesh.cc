/*
 * GenMesh.cc
 *
 *  Created on: Jun 4, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "GenMesh.hh"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "MyLegion.hh"
#include "Mesh.hh"
#include "Vec2.hh"
#include "InputFile.hh"

using namespace std;
using namespace Legion;

namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
    {
      TaskVariantRegistrar registrar(TID_GENPOINTS_RECT, "Gen Points Rect");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<GenMesh::genPointsRect>(registrar, "Gen Points Rect");
    }
    {
      TaskVariantRegistrar registrar(TID_GENPOINTS_PIE, "Gen Points Pie");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<GenMesh::genPointsPie>(registrar, "Gen Points Pie");
    }
    {
      TaskVariantRegistrar registrar(TID_GENPOINTS_HEX, "Gen Points Hex");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<GenMesh::genPointsHex>(registrar, "Gen Points Hex");
    }
    {
      TaskVariantRegistrar registrar(TID_GENZONES_PIE, "Gen Zones Pie");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<GenMesh::genZonesPie>(registrar, "Gen Zones Pie");
    }
    {
      TaskVariantRegistrar registrar(TID_GENSIDES_RECT, "Gen Sides Rect");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<GenMesh::genSidesRect>(registrar, "Gen Sides Rect");
    }
    {
      TaskVariantRegistrar registrar(TID_GENSIDES_PIE, "Gen Sides Pie");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<GenMesh::genSidesPie>(registrar, "Gen Sides Pie");
    }
    {
      TaskVariantRegistrar registrar(TID_GENSIDES_HEX, "Gen Sides Hex");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<GenMesh::genSidesHex>(registrar, "Gen Sides Hex");
    }
}
}; // namespace

GenMesh::GenMesh(const InputFile* inp) {
    meshtype = inp->getString("meshtype", "");
    if (meshtype.empty()) {
        cerr << "Error:  must specify meshtype" << endl;
        exit(1);
    }
    if (meshtype != "pie" &&
            meshtype != "rect" &&
            meshtype != "hex") {
        cerr << "Error:  invalid meshtype " << meshtype << endl;
        exit(1);
    }
    vector<double> params =
            inp->getDoubleList("meshparams", vector<double>());
    if (params.empty()) {
        cerr << "Error:  must specify meshparams" << endl;
        exit(1);
    }
    if (params.size() > 4) {
        cerr << "Error:  meshparams must have <= 4 values" << endl;
        exit(1);
    }

    nzx = params[0];
    nzy = (params.size() >= 2 ? params[1] : nzx);
    if (meshtype != "pie")
        lenx = (params.size() >= 3 ? params[2] : 1.0);
    else
        // convention:  x = theta, y = r
        lenx = (params.size() >= 3 ? params[2] : 90.0)
                * M_PI / 180.0;
    leny = (params.size() >= 4 ? params[3] : 1.0);

    if (nzx <= 0 || nzy <= 0 || lenx <= 0. || leny <= 0. ) {
        cerr << "Error:  meshparams values must be positive" << endl;
        exit(1);
    }
    if (meshtype == "pie" && lenx >= 2. * M_PI) {
        cerr << "Error:  meshparams theta must be < 360" << endl;
        exit(1);
    }
    pieces = false; // not initialized yet
}


GenMesh::~GenMesh() {}


void GenMesh::generate(
        const int numpcs,
        vector<double2>& pointpos,
        vector<int>& pointcolors,
        colormap& pointmcolors,
        vector<int>& zonestart,
        vector<int>& zonesize,
        vector<int>& zonepoints,
        vector<int>& zonecolors) {

    // do calculations common to all mesh types
    calcNumPieces(numpcs);
    zxbounds.push_back(-1);
    for (int pcx = 1; pcx < numpcx; ++pcx)
        zxbounds.push_back(pcx * nzx / numpcx);
    zxbounds.push_back(nzx + 1);
    zybounds.push_back(-1);
    for (int pcy = 1; pcy < numpcy; ++pcy)
        zybounds.push_back(pcy * nzy / numpcy);
    zybounds.push_back(nzy + 1);

    // mesh type-specific calculations
    if (meshtype == "pie")
        generatePie(pointpos, pointcolors, pointmcolors,
                zonestart, zonesize, zonepoints, zonecolors);
    else if (meshtype == "rect")
        generateRect(pointpos, pointcolors, pointmcolors,
                zonestart, zonesize, zonepoints, zonecolors);
    else if (meshtype == "hex")
        generateHex(pointpos, pointcolors, pointmcolors,
                zonestart, zonesize, zonepoints, zonecolors);
    else
      assert(false);
}


void GenMesh::generateRect(
        vector<double2>& pointpos,
        vector<int>& pointcolors,
        colormap& pointmcolors,
        vector<int>& zonestart,
        vector<int>& zonesize,
        vector<int>& zonepoints,
        vector<int>& zonecolors) {

    const int nz = nzx * nzy;
    const int npx = nzx + 1;
    const int npy = nzy + 1;
    const int np = npx * npy;

    // generate point coordinates
    pointpos.reserve(np);
    double dx = lenx / (double) nzx;
    double dy = leny / (double) nzy;
    int pcy = 0;
    for (int j = 0; j < npy; ++j) {
        if (j >= zybounds[pcy+1]) pcy += 1;
        double y = dy * (double) j;
        int pcx = 0;
        for (int i = 0; i < npx; ++i) {
            if (i >= zxbounds[pcx+1]) pcx += 1;
            double x = dx * (double) i;
            pointpos.push_back(make_double2(x, y));
            int c = pcy * numpcx + pcx;
            if (i != zxbounds[pcx] && j != zybounds[pcy])
                pointcolors.push_back(c);
            else {
                int p = pointpos.size() - 1;
                pointcolors.push_back(MULTICOLOR);
                vector<int>& pmc = pointmcolors[p];
                if (i == zxbounds[pcx] && j == zybounds[pcy])
                    pmc.push_back(c - numpcx - 1);
                if (j == zybounds[pcy]) pmc.push_back(c - numpcx);
                if (i == zxbounds[pcx]) pmc.push_back(c - 1);
                pmc.push_back(c);
            }
        }
    }

    // generate zone adjacency lists
    zonestart.reserve(nz);
    zonesize.reserve(nz);
    zonepoints.reserve(4 * nz);
    zonecolors.reserve(nz);
    pcy = 0;
    for (int j = 0; j < nzy; ++j) {
        if (j >= zybounds[pcy+1]) pcy += 1;
        int pcx = 0;
        for (int i = 0; i < nzx; ++i) {
            if (i >= zxbounds[pcx+1]) pcx += 1;
            zonestart.push_back(zonepoints.size());
            zonesize.push_back(4);
            int p0 = j * npx + i;
            zonepoints.push_back(p0);
            zonepoints.push_back(p0 + 1);
            zonepoints.push_back(p0 + npx + 1);
            zonepoints.push_back(p0 + npx);
            zonecolors.push_back(pcy * numpcx + pcx);
        }
    }

}


void GenMesh::generatePie(
        vector<double2>& pointpos,
        vector<int>& pointcolors,
        colormap& pointmcolors,
        vector<int>& zonestart,
        vector<int>& zonesize,
        vector<int>& zonepoints,
        vector<int>& zonecolors) {

    const int nz = nzx * nzy;
    const int npx = nzx + 1;
    const int npy = nzy + 1;
    const int np = npx * (npy - 1) + 1;

    // generate point coordinates
    pointpos.reserve(np);
    double dth = lenx / (double) nzx;
    double dr  = leny / (double) nzy;
    int pcy = 0;
    for (int j = 0; j < npy; ++j) {
        if (j >= zybounds[pcy+1]) pcy += 1;
        if (j == 0) {
            // special case:  "row" at origin only contains
            // one point, shared by all pieces in row
            pointpos.push_back(make_double2(0., 0.));
            if (numpcx == 1)
                pointcolors.push_back(0);
            else {
                pointcolors.push_back(MULTICOLOR);
                vector<int>& pmc = pointmcolors[0];
                for (int c = 0; c < numpcx; ++c)
                    pmc.push_back(c);
            }
            continue;
        }
        double r = dr * (double) j;
        int pcx = 0;
        for (int i = 0; i < npx; ++i) {
            if (i >= zxbounds[pcx+1]) pcx += 1;
            double th = dth * (double) (nzx - i);
            double x = r * cos(th);
            double y = r * sin(th);
            pointpos.push_back(make_double2(x, y));
            int c = pcy * numpcx + pcx;
            if (i != zxbounds[pcx] && j != zybounds[pcy])
                pointcolors.push_back(c);
            else {
                int p = pointpos.size() - 1;
                pointcolors.push_back(MULTICOLOR);
                vector<int>& pmc = pointmcolors[p];
                if (i == zxbounds[pcx] && j == zybounds[pcy])
                    pmc.push_back(c - numpcx - 1);
                if (j == zybounds[pcy]) pmc.push_back(c - numpcx);
                if (i == zxbounds[pcx]) pmc.push_back(c - 1);
                pmc.push_back(c);
            }
        }
    }

    // generate zone adjacency lists
    zonestart.reserve(nz);
    zonesize.reserve(nz);
    zonepoints.reserve(4 * nz);
    zonecolors.reserve(nz);
    pcy = 0;
    for (int j = 0; j < nzy; ++j) {
        if (j >= zybounds[pcy+1]) pcy += 1;
        int pcx = 0;
        for (int i = 0; i < nzx; ++i) {
            if (i >= zxbounds[pcx+1]) pcx += 1;
            zonestart.push_back(zonepoints.size());
            int p0 = j * npx + i - (npx - 1);
            if (j == 0) {
                zonesize.push_back(3);
                zonepoints.push_back(0);
            }
            else {
                zonesize.push_back(4);
                zonepoints.push_back(p0);
                zonepoints.push_back(p0 + 1);
            }
            zonepoints.push_back(p0 + npx + 1);
            zonepoints.push_back(p0 + npx);
            zonecolors.push_back(pcy * numpcx + pcx);
        }
    }

}


void GenMesh::generateHex(
        vector<double2>& pointpos,
        vector<int>& pointcolors,
        colormap& pointmcolors,
        vector<int>& zonestart,
        vector<int>& zonesize,
        vector<int>& zonepoints,
        vector<int>& zonecolors) {

    const int nz = nzx * nzy;
    const int npx = nzx + 1;
    const int npy = nzy + 1;

    // generate point coordinates
    pointpos.resize(2 * npx * npy);  // upper bound
    double dx = lenx / (double) (nzx - 1);
    double dy = leny / (double) (nzy - 1);

    vector<int> pbase(npy);
    int p = 0;
    int pcy = 0;
    for (int j = 0; j < npy; ++j) {
        if (j >= zybounds[pcy+1]) pcy += 1;
        pbase[j] = p;
        double y = dy * ((double) j - 0.5);
        y = max(0., min(leny, y));
        int pcx = 0;
        for (int i = 0; i < npx; ++i) {
            if (i >= zxbounds[pcx+1]) pcx += 1;
            double x = dx * ((double) i - 0.5);
            x = max(0., min(lenx, x));
            int c = pcy * numpcx + pcx;
            if (i == 0 || i == nzx || j == 0 || j == nzy) {
                pointpos[p++] = make_double2(x, y);
                if (i != zxbounds[pcx] && j != zybounds[pcy])
                    pointcolors.push_back(c);
                else {
                    int p1 = p - 1;
                    pointcolors.push_back(MULTICOLOR);
                    vector<int>& pmc = pointmcolors[p1];
                    if (j == zybounds[pcy]) pmc.push_back(c - numpcx);
                    if (i == zxbounds[pcx]) pmc.push_back(c - 1);
                    pmc.push_back(c);
                }
            }
            else {
                pointpos[p++] = make_double2(x - dx / 6., y + dy / 6.);
                pointpos[p++] = make_double2(x + dx / 6., y - dy / 6.);
                if (i != zxbounds[pcx] && j != zybounds[pcy]) {
                    pointcolors.push_back(c);
                    pointcolors.push_back(c);
                }
                else {
                    int p1 = p - 2;
                    int p2 = p - 1;
                    pointcolors.push_back(MULTICOLOR);
                    pointcolors.push_back(MULTICOLOR);
                    vector<int>& pmc1 = pointmcolors[p1];
                    vector<int>& pmc2 = pointmcolors[p2];
                    if (i == zxbounds[pcx] && j == zybounds[pcy]) {
                        pmc1.push_back(c - numpcx - 1);
                        pmc2.push_back(c - numpcx - 1);
                        pmc1.push_back(c - 1);
                        pmc2.push_back(c - numpcx);
                    }
                    else if (j == zybounds[pcy]) {
                        pmc1.push_back(c - numpcx);
                        pmc2.push_back(c - numpcx);
                    }
                    else {  // i == zxbounds[pcx]
                        pmc1.push_back(c - 1);
                        pmc2.push_back(c - 1);
                    }
                    pmc1.push_back(c);
                    pmc2.push_back(c);
                }
            }
        } // for i
    } // for j
    int np = p;
    pointpos.resize(np);

    // generate zone adjacency lists
    zonestart.resize(nz);
    zonesize.resize(nz);
    zonepoints.reserve(6 * nz);  // upper bound
    zonecolors.reserve(nz);
    pcy = 0;
    for (int j = 0; j < nzy; ++j) {
        if (j >= zybounds[pcy+1]) pcy += 1;
        int pbasel = pbase[j];
        int pbaseh = pbase[j+1];
        int pcx = 0;
        for (int i = 0; i < nzx; ++i) {
            if (i >= zxbounds[pcx+1]) pcx += 1;
            int z = j * nzx + i;
            vector<int> v(6);
            v[1] = pbasel + 2 * i;
            v[0] = v[1] - 1;
            v[2] = v[1] + 1;
            v[5] = pbaseh + 2 * i;
            v[4] = v[5] + 1;
            v[3] = v[4] + 1;
            if (j == 0) {
                v[0] = pbasel + i;
                v[2] = v[0] + 1;
                if (i == nzx - 1) v.erase(v.begin()+3);
                v.erase(v.begin()+1);
            } // if j
            else if (j == nzy - 1) {
                v[5] = pbaseh + i;
                v[3] = v[5] + 1;
                v.erase(v.begin()+4);
                if (i == 0) v.erase(v.begin()+0);
            } // else if j
            else if (i == 0)
                v.erase(v.begin()+0);
            else if (i == nzx - 1)
                v.erase(v.begin()+3);
            zonestart[z] = zonepoints.size();
            zonesize[z] = v.size();
            zonepoints.insert(zonepoints.end(), v.begin(), v.end());
            zonecolors.push_back(pcy * numpcx + pcx);
        } // for i
    } // for j

}


void GenMesh::calcNumPieces(const int numpcs) {

    // If we already computed this then we don't need to do it again
    if (pieces) return;
    // pick numpcx, numpcy such that pieces are as close to square
    // as possible
    // we would like:  nzx / numpcx == nzy / numpcy,
    // where numpcx * numpcy = numpcs (total number of pieces)
    // this solves to:  numpcx = sqrt(numpcs * nzx / nzy)
    // we compute this, assuming nzx <= nzy (swap if necessary)
    double nx = static_cast<double>(nzx);
    double ny = static_cast<double>(nzy);
    bool swapflag = (nx > ny);
    if (swapflag) swap(nx, ny);
    double n = sqrt(numpcs * nx / ny);
    // need to constrain n to be an integer with numpcs % n == 0
    // try rounding n both up and down
    int n1 = floor(n + 1.e-12);
    n1 = max(n1, 1);
    while (numpcs % n1 != 0) --n1;
    int n2 = ceil(n - 1.e-12);
    while (numpcs % n2 != 0) ++n2;
    // pick whichever of n1 and n2 gives blocks closest to square,
    // i.e. gives the shortest long side
    double longside1 = max(nx / n1, ny / (numpcs/n1));
    double longside2 = max(nx / n2, ny / (numpcs/n2));
    numpcx = (longside1 <= longside2 ? n1 : n2);
    numpcy = numpcs / numpcx;
    if (swapflag) swap(numpcx, numpcy);
    // Record that we computed this
    pieces = true;
}

int GenMesh::calcNumPoints(const int numpcs) {
    // First calculate the number of pieces if we haven't done so yet
    calcNumPieces(numpcs);
    if (meshtype == "rect") {
      const int npx = nzx + 1;
      const int npy = nzy + 1;
      const int np = npx * npy;
      return np;
    } else if (meshtype == "pie") {
      const int npx = nzx + 1;
      const int npy = nzy + 1;
      const int np = npx * (npy - 1) + 1;
      return np;
    } else if (meshtype == "hex") {
      const int npx = nzx + 1;
      const int npy = nzy + 1;
      const int np = 2 * npx * npy;
      return np;
    } else {
      assert(false);
      return -1;
    }
}

int GenMesh::calcNumZones(const int numpcs) {
    // First calculate the number of pieces if we haven't done so yet
    calcNumPieces(numpcs);
    if (meshtype == "rect") {
      const int nz = nzx * nzy;
      return nz;
    } else if (meshtype == "pie") {
      const int nz = nzx * nzy;
      return nz;
    } else if (meshtype == "hex") {
      const int nz = nzx * nzy;
      return nz;
    } else {
      assert(false);
      return -1;
    }
}

int GenMesh::calcNumSides(const int numpcs) {
    if (meshtype == "rect") {
      const int nz = calcNumZones(numpcs);
      return 4 * nz;
    } else if (meshtype == "pie") {
      calcNumPieces(numpcs);
      // 4 sides on most of the zones, but only 3 on the inner ones near the origin
      return 4 * nzx * (nzy - 1) + 3 * nzx; 
    } else if (meshtype == "hex") {
      const int nz = calcNumZones(numpcs);
      return 6 * nz;
    } else {
      assert(false);
      return -1;
    }
}

void GenMesh::generatePointsParallel(
            const int numpcs,
            Runtime *runtime,
            Context ctx,
            LogicalRegion points_lr,
            LogicalPartition points_lp,
            IndexSpace piece_is)
{
    // Have to do this before getting the domain
    calcNumPieces(numpcs);
    const GenPointArgs args(this);
    RegionRequirement req(points_lp, 0/*identity projection*/,
                          WRITE_DISCARD, EXCLUSIVE, points_lr);
    req.add_field(FID_PX);
    req.add_field(FID_PIECE);
    if (meshtype == "rect") {
      IndexTaskLauncher launcher(TID_GENPOINTS_RECT, piece_is,
          TaskArgument(&args, sizeof(args)), ArgumentMap());
      launcher.add_region_requirement(req);
      runtime->execute_index_space(ctx, launcher);
    } else if (meshtype == "pie") {
      IndexTaskLauncher launcher(TID_GENPOINTS_PIE, piece_is,
          TaskArgument(&args, sizeof(args)), ArgumentMap());
      launcher.add_region_requirement(req);
      runtime->execute_index_space(ctx, launcher);
    } else if (meshtype == "hex") {
      IndexTaskLauncher launcher(TID_GENPOINTS_HEX, piece_is,
          TaskArgument(&args, sizeof(args)), ArgumentMap());
      launcher.add_region_requirement(req);
      runtime->execute_index_space(ctx, launcher);
    } else {
      assert(false);
    }
}

void GenMesh::generateZonesParallel(
            const int numpcs,
            Runtime *runtime,
            Context ctx,
            LogicalRegion zones_lr,
            LogicalPartition zones_lp,
            IndexSpace piece_is)
{
    calcNumPieces(numpcs);
    

    if (meshtype == "rect") {
      const int num_points = 4;
      FillLauncher fill(zones_lr, zones_lr, 
          TaskArgument(&num_points, sizeof(num_points)));
      fill.add_field(FID_ZNUMP);
      runtime->fill_fields(ctx, fill);
    } else if (meshtype == "pie") {
      RegionRequirement req(zones_lp, 0/*identity projection*/,
                          WRITE_DISCARD, EXCLUSIVE, zones_lr);
      req.add_field(FID_ZNUMP);
      const GenZoneArgs args(this);
      IndexTaskLauncher launcher(TID_GENZONES_PIE, piece_is,
          TaskArgument(&args, sizeof(args)), ArgumentMap());
      launcher.add_region_requirement(req);
      runtime->execute_index_space(ctx, launcher);
    } else if (meshtype == "hex") {
      const int num_points = 6;
      FillLauncher fill(zones_lr, zones_lr, 
          TaskArgument(&num_points, sizeof(num_points)));
      fill.add_field(FID_ZNUMP);
      runtime->fill_fields(ctx, fill);
    } else {
      assert(false);
    }
}


void GenMesh::generateSidesParallel(
            const int numpcs,
            Runtime *runtime,
            Context ctx,
            LogicalRegion sides_lr,
            LogicalPartition sides_lp,
            IndexSpace piece_is)
{
    // Have to do this before getting the domain
    calcNumPieces(numpcs);
    const GenSideArgs args(this);
    RegionRequirement req(sides_lp, 0/*identity projection*/,
                          WRITE_DISCARD, EXCLUSIVE, sides_lr);
    // Only fill in the load pointers here
    req.add_field(FID_MAPSP1TEMP);
    req.add_field(FID_MAPSP2TEMP);
    req.add_field(FID_MAPSZ);
    req.add_field(FID_MAPSS3);
    req.add_field(FID_MAPSS4);
    req.add_field(FID_PIECE);
    if (meshtype == "rect") {
      IndexTaskLauncher launcher(TID_GENSIDES_RECT, piece_is,
          TaskArgument(&args, sizeof(args)), ArgumentMap());
      launcher.add_region_requirement(req);
      runtime->execute_index_space(ctx, launcher);
    } else if (meshtype == "pie") {
      IndexTaskLauncher launcher(TID_GENSIDES_PIE, piece_is,
          TaskArgument(&args, sizeof(args)), ArgumentMap());
      launcher.add_region_requirement(req);
      runtime->execute_index_space(ctx, launcher);
    } else if (meshtype == "hex") {
      IndexTaskLauncher launcher(TID_GENSIDES_HEX, piece_is,
          TaskArgument(&args, sizeof(args)), ArgumentMap());
      launcher.add_region_requirement(req);
      runtime->execute_index_space(ctx, launcher);
    } else {
      assert(false);
    }
}

void GenMesh::genPointsRect(
            const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx,
            Runtime *runtime)
{
  const GenPointArgs *args = reinterpret_cast<const GenPointArgs*>(task->args);
  const int npx = args->nzx + 1;

  const double dx = args->lenx / (double) args->nzx;
  const double dy = args->leny / (double) args->nzy;

  const int zones_per_piecex = (args->nzx + args->numpcx - 1) / args->numpcx;
  const int zones_per_piecey = (args->nzy + args->numpcy - 1) / args->numpcy;

  const IndexSpace &isp = task->regions[0].region.get_index_space();
  const AccessorWD<double2> acc_px(regions[0], FID_PX);
  const AccessorWD<coord_t> acc_piece(regions[0], FID_PIECE);
  for (PointIterator itr(runtime, isp); itr(); itr++)
  {
    // Figure out the physical location based on the logical ID
    const int i = itr[0] % npx;
    const int j = itr[0] / npx;
    const double x = dx * double(i);
    const double y = dy * double(j);
    acc_px[*itr] = make_double2(x, y);
    // Tile the mesh so pieces are dense rectangles 
    // Boundary zones will own a few extra points
    const int piecex = ((i == args->nzx) ? i-1 : i) / zones_per_piecex;
    assert(piecex < args->numpcx);
    const int piecey = ((j == args->nzy) ? j-1 : j) / zones_per_piecey;
    assert(piecey < args->numpcy);
    acc_piece[*itr] = piecey * args->numpcx + piecex;
  }
}

void GenMesh::genPointsPie(
            const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx,
            Runtime *runtime)
{
  const GenPointArgs *args = reinterpret_cast<const GenPointArgs*>(task->args);
  const int npx = args->nzx + 1;

  const double dth = args->lenx / (double) args->nzx;
  const double dr  = args->leny / (double) args->nzy;

  const int zones_per_piecex = (args->nzx + args->numpcx - 1) / args->numpcx;
  const int zones_per_piecey = (args->nzy + args->numpcy - 1) / args->numpcy;

  const IndexSpace &isp = task->regions[0].region.get_index_space();
  const AccessorWD<double2> acc_px(regions[0], FID_PX);
  const AccessorWD<coord_t> acc_piece(regions[0], FID_PIECE);
  for (PointIterator itr(runtime, isp); itr(); itr++)
  {
    // Figure out the physical location based on the logical ID
    if (itr[0] == 0) {
      // Special case for the origin
      acc_px[*itr] = make_double2(0., 0.);
      acc_piece[*itr] = 0;
    } else {
      const int i = (itr[0]-1) % npx;
      const int j = (itr[0]-1) / npx + 1;
      const double th = dth * (double)(args->nzx - i);
      const double r = dr * (double) j;
      const double x = r * cos(th);
      const double y = r * sin(th);
      acc_px[*itr] = make_double2(x, y);
      // Tile the mesh so pieces are dense rectangles 
      // Boundary zones will own a few extra points
      const int piecex = ((i == args->nzx) ? i-1 : i) / zones_per_piecex;
      assert(piecex < args->numpcx);
      const int piecey = ((j == args->nzy) ? j-1 : j) / zones_per_piecey;
      assert(piecey < args->numpcy);
      acc_piece[*itr] = piecey * args->numpcx + piecex;
    }
  }
}

void GenMesh::genPointsHex(
            const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx,
            Runtime *runtime)
{
#if 0
  const GenPointArgs *args = reinterpret_cast<const GenPointArgs*>(task->args);
  const int npx = args->nzx + 1;

  const double dx = args->lenx / (double) (args->nzx - 1);
  const double dy = args->leny / (double) (args->nzy - 1);

  const IndexSpace &isp = task->regions[0].region.get_index_space();
  const AccessorWD<double2> acc_px(regions[0], FID_PX);
  for (PointIterator itr(runtime, isp); itr(); itr++)
  {
  }
#else
  assert(false); // TODO
#endif
}

void GenMesh::genZonesPie(
            const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx,
            Runtime *runtime)
{
  const GenZoneArgs *args = reinterpret_cast<const GenZoneArgs*>(task->args);

  const int zones_per_piecex = (args->nzx + args->numpcx - 1) / args->numpcx;
  const int zones_per_piecey = (args->nzy + args->numpcy - 1) / args->numpcy;

  const IndexSpace &isz = task->regions[0].region.get_index_space();
  const AccessorWD<int> acc_nump(regions[0], FID_ZNUMP);
  for (PointIterator itr(runtime, isz); itr(); itr++)
  {
    const int piecey = itr[0] / (zones_per_piecey * args->nzx);
    assert(piecey < args->numpcy);
    const int remainder = itr[0] % (zones_per_piecey * args->nzx);

    const int local_zones_per_piecey = 
      piecey < (args->numpcy-1) ? zones_per_piecey : // not the last
        ((args->nzy % zones_per_piecey) == 0) ? // last so see if evenly divisible
          zones_per_piecey : args->nzy % zones_per_piecey;
    const int zones_per_row_piece = local_zones_per_piecey * zones_per_piecex;

    const int piecex = remainder / zones_per_row_piece;
    assert(piecex < args->numpcx);
    const int piece_zone = remainder % zones_per_row_piece;

    const int local_zones_per_piecex =
      piecex < (args->numpcx-1) ? zones_per_piecex : // not the last
        ((args->nzx % zones_per_piecex) == 0) ? // last so see if evenly divisible 
          zones_per_piecex : args->nzx % zones_per_piecex;

    const int localy = piece_zone / local_zones_per_piecex;
    const int zidy = piecey * zones_per_piecey + localy;

    // Three points if it is at the bottom, otherwise four
    acc_nump[*itr] = (zidy == 0) ? 3 : 4;
  }
}

void GenMesh::genSidesRect(
            const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx,
            Runtime *runtime)
{
  const GenSideArgs *args = reinterpret_cast<const GenSideArgs*>(task->args);

  const int zones_per_piecex = (args->nzx + args->numpcx - 1) / args->numpcx;
  const int zones_per_piecey = (args->nzy + args->numpcy - 1) / args->numpcy;
  
  const IndexSpace &iss = task->regions[0].region.get_index_space();
  const AccessorWD<Pointer> acc_sp1(regions[0], FID_MAPSP1TEMP);
  const AccessorWD<Pointer> acc_sp2(regions[0], FID_MAPSP2TEMP);
  const AccessorWD<Pointer> acc_sz(regions[0], FID_MAPSZ);
  const AccessorWD<Pointer> acc_ss3(regions[0], FID_MAPSS3);
  const AccessorWD<Pointer> acc_ss4(regions[0], FID_MAPSS4);
  const AccessorWD<Pointer> acc_piece(regions[0], FID_PIECE);
  for (PointIterator itr(runtime, iss); itr(); itr++)
  {
    // First we can compute our side pointers since they are easy
    const int side = itr[0] % 4;
    if (side == 3)
      acc_ss3[*itr] = *itr - Pointer(3);
    else
      acc_ss3[*itr] = *itr + Pointer(1);
    if (side == 0)
      acc_ss4[*itr] = *itr + Pointer(3);
    else
      acc_ss4[*itr] = *itr - Pointer(1);
    // Now figure out which zone we're a part of
    // zones for a piece are all contiguous
    const int zone = itr[0] / 4; // 4 sides per zone
    acc_sz[*itr] = zone;
    // Get the simulation x-y coorindate of our zone
    // This is tricky since there can be truncated zones on the edges
    const int piecey = zone / (zones_per_piecey * args->nzx);
    assert(piecey < args->numpcy);
    const int remainder = zone % (zones_per_piecey * args->nzx);

    const int local_zones_per_piecey = 
      piecey < (args->numpcy-1) ? zones_per_piecey : // not the last
        ((args->nzy % zones_per_piecey) == 0) ? // last so see if evently divisible
          zones_per_piecey : args->nzy % zones_per_piecey;
    const int zones_per_row_piece = local_zones_per_piecey * zones_per_piecex;

    const int piecex = remainder / zones_per_row_piece;
    assert(piecex < args->numpcx);
    acc_piece[*itr] = piecey * args->numpcx + piecex;
    const int piece_zone = remainder % zones_per_row_piece; 
    // Now figure out the local zone count in the x dimension
    const int local_zones_per_piecex = 
      piecex < (args->numpcx-1) ? zones_per_piecex : // not the last
        ((args->nzx % zones_per_piecex) == 0) ? // last so see if evenly divisible
          zones_per_piecex : args->nzx % zones_per_piecex;

    const int localx = piece_zone % local_zones_per_piecex; 
    const int localy = piece_zone / local_zones_per_piecex;

    const int zidx = piecex * zones_per_piecex + localx;
    assert(zidx < args->nzx);
    const int zidy = piecey * zones_per_piecey + localy; 
    assert(zidy < args->nzy);
    // Last we can figure out the indexes for our points
    int pidx1, pidx2, pidy1, pidy2;
    switch (side)
    {
      case 0:
        {
          pidx1 = zidx;
          pidx2 = zidx + 1;
          pidy1 = zidy;
          pidy2 = zidy;
          break;
        }
      case 1:
        {
          pidx1 = zidx + 1;
          pidx2 = zidx + 1;
          pidy1 = zidy;
          pidy2 = zidy + 1;
          break;
        }
      case 2:
        {
          pidx1 = zidx + 1;
          pidx2 = zidx;
          pidy1 = zidy + 1;
          pidy2 = zidy + 1;
          break;
        }
      case 3:
        {
          pidx1 = zidx;
          pidx2 = zidx;
          pidy1 = zidy + 1;
          pidy2 = zidy;
          break;
        }
      default:
        assert(false);
    }
    acc_sp1[*itr] = Pointer(pidy1 * (args->nzx + 1) + pidx1);
    assert(acc_sp1[*itr][0] < 100);
    acc_sp2[*itr] = Pointer(pidy2 * (args->nzx + 1) + pidx2);
    assert(acc_sp2[*itr][0] < 100);
  }
}

void GenMesh::genSidesPie(
            const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx,
            Runtime *runtime)
{
  const GenSideArgs *args = reinterpret_cast<const GenSideArgs*>(task->args);

  const int zones_per_piecex = (args->nzx + args->numpcx - 1) / args->numpcx;
  const int zones_per_piecey = (args->nzy + args->numpcy - 1) / args->numpcy;
  const int npx = args->nzx + 1;
  
  const IndexSpace &iss = task->regions[0].region.get_index_space();
  const AccessorWD<Pointer> acc_sp1(regions[0], FID_MAPSP1TEMP);
  const AccessorWD<Pointer> acc_sp2(regions[0], FID_MAPSP2TEMP);
  const AccessorWD<Pointer> acc_sz(regions[0], FID_MAPSZ);
  const AccessorWD<Pointer> acc_ss3(regions[0], FID_MAPSS3);
  const AccessorWD<Pointer> acc_ss4(regions[0], FID_MAPSS4);
  const AccessorWD<Pointer> acc_piece(regions[0], FID_PIECE);
  for (PointIterator itr(runtime, iss); itr(); itr++)
  {
    // First figure out our piece and zone from our side
    int piecex=-1, piecey=-1, piece_zone=-1, piece_zonex=-1, piece_zoney=-1, side=-1;
    // We'll find this a dumb way for now
    bool found = false;
    int current_side = 0;
    const int target_side = itr[0];
    for (int j1 = 0; (j1 < args->numpcy) && !found; j1++)
    {
      const int local_zones_per_piecey = (j1 < (args->numpcy-1)) ? zones_per_piecey :
        ((args->nzy % zones_per_piecey) == 0) ? // last so see if divisible
          zones_per_piecey : args->nzy % zones_per_piecey;
      
      for (int i1 = 0; (i1 < args->numpcx) && !found; i1++)
      {
        const int local_zones_per_piecex = (i1 < (args->numpcx-1)) ? zones_per_piecex :
          ((args->nzx % zones_per_piecex) == 0) ? // last so see if divisible
            zones_per_piecex : args->nzx % zones_per_piecex;
        // See how many sides we have in this piece
        const int sides_in_this_piece = (j1 == 0) ? 
          3 * local_zones_per_piecex + 4 * (local_zones_per_piecey - 1) * local_zones_per_piecex : 
          4 * local_zones_per_piecex * local_zones_per_piecey;
        
        if ((current_side + sides_in_this_piece) <= target_side)
          current_side += sides_in_this_piece;
        else
        {
          // We've found the piece
          piecex = i1;
          piecey = j1;
          // Now we go looking for the zone in the piece
          for (int j2 = 0; (j2 < local_zones_per_piecey) && !found; j2++)
          {
            for (int i2 = 0; (i2 < local_zones_per_piecex) && !found; i2++)
            {
              if ((j1 == 0) && (j2 == 0))
              {
                if ((current_side + 3) <= target_side)
                  current_side += 3;
                else
                {
                  // Found the zone
                  piece_zonex = i2;
                  piece_zoney = j2;
                  piece_zone = j2 * local_zones_per_piecex + i2;
                  side = (target_side - current_side) % 3;
                  found = true;
                }
              }
              else
              {
                if ((current_side + 4) <= target_side)
                  current_side += 4;
                else
                {
                  // Found the zone
                  piece_zonex = i2;
                  piece_zoney = j2;
                  piece_zone = j2 * local_zones_per_piecex + i2;
                  side = (target_side - current_side) % 4;
                  found = true;
                }
              }
            }
          }
          assert(found);
        }
      }
    }
    assert(found);
    // Now we can assign our zone pointer
    const int local_zones_per_piecey = 
      piecey < (args->numpcy-1) ? zones_per_piecey : // not the last
        ((args->nzy % zones_per_piecey) == 0) ? // last so see if evenly divisible
          zones_per_piecey : args->nzy % zones_per_piecey;
    const int zones_per_row_piece = local_zones_per_piecey * zones_per_piecex;
    acc_sz[*itr] = piecey * zones_per_piecey * args->nzx + 
                    piecex * zones_per_row_piece + piece_zone;
    acc_piece[*itr] = piecey * args->numpcx + piecex;
    // Then figure out our zone coordinates
    const int zidx = piecex * zones_per_piecex + piece_zonex;
    const int zidy = piecey * zones_per_piecey + piece_zoney; 
    // Do different things for sides from inner most zones from sides for other zones
    if (zidy == 0)
    {
      // Side maps are all just local address from here
      if (side == 2)
        acc_ss3[*itr] = *itr - Pointer(2);
      else
        acc_ss3[*itr] = *itr + Pointer(1);
      if (side == 0)
        acc_ss4[*itr] = *itr + Pointer(2);
      else
        acc_ss4[*itr] = *itr - Pointer(1);
      const int p0 = zidx + 1;
      switch (side)
      {
        case 0:
          {
            acc_sp1[*itr] = 0;
            acc_sp2[*itr] = p0 + 1;
            break;
          }
        case 1:
          {
            acc_sp1[*itr] = p0 + 1;
            acc_sp2[*itr] = p0;
            break;
          }
        case 2:
          {
            acc_sp1[*itr] = p0;
            acc_sp2[*itr] = 0;
            break;
          }
        default:
          assert(false);
      }
    }
    else
    {
      // Side maps are all just local address from here
      if (side == 3)
        acc_ss3[*itr] = *itr - Pointer(3);
      else
        acc_ss3[*itr] = *itr + Pointer(1);
      if (side == 0)
        acc_ss4[*itr] = *itr + Pointer(3);
      else
        acc_ss4[*itr] = *itr - Pointer(1);
      const int p0 = (zidy - 1) * npx + zidx + 1;
      switch (side)
      {
        case 0:
          {
            acc_sp1[*itr] = p0;
            acc_sp2[*itr] = p0 + 1;
            break;
          }
        case 1:
          {
            acc_sp1[*itr] = p0 + 1;
            acc_sp2[*itr] = p0 + npx + 1;
            break;
          }
        case 2:
          {
            acc_sp1[*itr] = p0 + npx + 1;
            acc_sp2[*itr] = p0 + npx;
            break;
          }
        case 3:
          {
            acc_sp1[*itr] = p0 + npx;
            acc_sp2[*itr] = p0;
            break;
          }
        default:
          assert(false);
      }
    }
  }
}

void GenMesh::genSidesHex(
            const Task *task,
            const std::vector<PhysicalRegion> &regions,
            Context ctx,
            Runtime *runtime)
{
  assert(false); // TODO
}

