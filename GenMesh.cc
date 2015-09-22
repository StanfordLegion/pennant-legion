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

#include "Vec2.hh"
#include "InputFile.hh"

using namespace std;


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

}

