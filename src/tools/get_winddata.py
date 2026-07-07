#!/usr/bin/env python
import cdsapi

c = cdsapi.Client()

"""
https://confluence.ecmwf.int/plugins/viewsource/viewpagesrc.action?pageId=158636068

Copyright 2023 ECMWF.

This software is licensed under the terms of the Apache Licence Version 2.0
which can be obtained at http://www.apache.org/licenses/LICENSE-2.0

In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation
nor does it submit to any jurisdiction.
"""

# data download specifications:
cls = "ea"  # do not change
expver = "1"  # do not change
levtype = "ml"  # do not change
stream = "oper"  # do not change
# date: Specify a single date as "2018-01-01" or a period as
# "2018-08-01/to/2018-01-31". For periods > 1 month see
# https://confluence.ecmwf.int/x/l7GqB
date = "2026-06-05"
# request type: Use "an" (analysis) unless you have a particular reason to
# use "fc" (forecast).
tp = "an"
# time: ERA5 data is hourly. Specify a single time as "00:00:00", or a
# range as "00:00:00/01:00:00/02:00:00" or "00:00:00/to/23:00:00/by/1".
time = "13:00:00/to/15:00:00/by/1"

# All 137 model levels, as the slash-separated string the CDS API expects.
levelist_all = "/".join(str(lvl) for lvl in range(1, 138))


c.retrieve(
    "reanalysis-era5-complete",
    {
        "class": cls,
        "date": date,
        "expver": expver,
        # Geopotential (z) and Logarithm of surface pressure (lnsp) are 2D
        # fields, archived as model level 1
        "levelist": "1",
        "levtype": levtype,
        # Geopotential (z) and Logarithm of surface pressure (lnsp)
        "param": "129/152",
        "stream": stream,
        "time": time,
        "type": tp,
        # Latitude/longitude grid: east-west (longitude) and north-south
        # resolution (latitude). Default: 0.25 x 0.25
        "grid": [0.25, 0.25],
        # North, West, South, East. Default: global.
        # example: [60, -10, 50, 2]
        "area": [45, -80, 30, -70],
    },
    "zlnsp_ml.grib",
)
