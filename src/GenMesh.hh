/*
 * GenMesh.hh
 *
 *  Created on: Jun 4, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef GENMESH_HH_
#define GENMESH_HH_

#include <string>
#include <vector>
#include <map>
#include "Vec2.hh"

// forward declarations
class InputFile;


typedef std::map<int, std::vector<int> > colormap;

const int MULTICOLOR = -1;

class GenMesh {
public:

    std::string meshtype;       // generated mesh type
    int nzx, nzy;               // number of zones, in x and y
                                // directions
    double lenx, leny;          // length of mesh sides, in x and y
                                // directions
    int numpcx, numpcy;         // number of pieces to generate,
                                // in x and y directions
    std::vector<int> zxbounds, zybounds;
                                // boundaries of pieces, in x and y
                                // directions

    GenMesh(const InputFile* inp);
    ~GenMesh();

    void generate(
            const int numpcs,
            std::vector<double2>& pointpos,
            std::vector<int>& pointcolors,
            colormap& pointmcolors,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& zonecolors);

    void generateRect(
            std::vector<double2>& pointpos,
            std::vector<int>& pointcolors,
            colormap& pointmcolors,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& zonecolors);

    void generatePie(
            std::vector<double2>& pointpos,
            std::vector<int>& pointcolors,
            colormap& pointmcolors,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& zonecolors);

    void generateHex(
            std::vector<double2>& pointpos,
            std::vector<int>& pointcolors,
            colormap& pointmcolors,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& zonecolors);

    void calcNumPieces(const int numpc);

}; // class GenMesh


#endif /* GENMESH_HH_ */
