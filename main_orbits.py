#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate figures for various orbit generators.

This script is used by Gitlab-CI to generate example figures.

Authors:
    Jean-Baptiste Bayle <j2b.bayle@gmail.com>
"""

import logging
import argparse
import lisaorbits



def main():
    """Main function."""
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger('lisaorbits').setLevel(logging.INFO)

    generators = {
        'static-constellation': static_constellation,
        'keplerian-orbits': keplerian_orbits,
        'equalarmlength-orbits': equalarmlength_orbits,
        'esa-trailing-orbits': esa_trailing_orbits,
        'esa-leading-orbits': esa_leading_orbits,
        'readOEMOrbits': readOEMOrbits
    }

    parser = argparse.ArgumentParser(description='Orbit App')
    parser.add_argument('generator', help='type of orbit generator', choices=generators.keys())
    args = parser.parse_args()

    args.generator = 'readOEMOrbits'
    orbits = generators[args.generator]()
    # orbits = generators['readOEMOrbits']()
    orbits.write(f'{args.generator}.h5')
    orbits.plot_spacecraft(1, f'{args.generator}-sc1.pdf')
    orbits.plot_spacecraft(2, f'{args.generator}-sc2.pdf')
    orbits.plot_spacecraft(3, f'{args.generator}-sc3.pdf')
    orbits.plot_links(f'{args.generator}-links.pdf')


def static_constellation():
    """Generation `StaticConstellation` plot."""
    keplerian = keplerian_orbits()
    return lisaorbits.StaticConstellation.from_orbits(keplerian)


def keplerian_orbits():
    """Generate `KeplerianOrbits` plot."""
    return lisaorbits.KeplerianOrbits()


def equalarmlength_orbits():
    """Generate `EqualArmlengthOrbits` plot."""
    return lisaorbits.EqualArmlengthOrbits()


def esa_trailing_orbits():
    """Generate `OEMOrbits` plot from trailing ESA orbits."""
    return lisaorbits.OEMOrbits.from_included('esa-trailing')


def esa_leading_orbits():
    """Generate `OEMOrbits` plot from leading ESA orbits."""
    return lisaorbits.OEMOrbits.from_included('esa-leading')

def readOEMOrbits():
    return lisaorbits.OEMOrbits(oem_1= './tq3/TianQinOrbit_120days_UTC/FullForce100935km_85.7deg-EphemerisFile_SC1-50s.oem',
                                oem_2= './tq3/TianQinOrbit_120days_UTC/FullForce100935km_85.7deg-EphemerisFile_SC2-50s.oem',
                                oem_3= './tq3/TianQinOrbit_120days_UTC/FullForce100935km_85.7deg-EphemerisFile_SC3-50s.oem')




if __name__ == '__main__':
    main()
