from lisainstrument import Instrument
import importlib_metadata
import get_my_conf as conf_var
from get_my_conf import *


def main(flag=1):
    filename_path = './lisa_dev.ini'
    get_config(filename_path)

    if flag == 1:
        instrument = Instrument(dt=0.25, size=10000, orbits='../testorbits/tq3/readOEMOrbits.h5')
    else:
        instrument = Instrument(conf_var.global_size, conf_var.global_dt, conf_var.global_t0,
                                conf_var.global_physics_upsampling, \
                                conf_var.global_aafilter, conf_var.global_telemetry_downsampling, \
                                conf_var.global_initial_telemetry_size, conf_var.global_orbits,
                                conf_var.global_orbit_dataset, conf_var.global_gws,
                                conf_var.global_interpolation, conf_var.global_glitches, \
                                conf_var.global_lock, conf_var.global_fplan, conf_var.global_laser_asds,
                                conf_var.global_laser_shape, conf_var.global_central_freq,
                                conf_var.global_offset_freqs, conf_var.global_modulation_asds, \
                                conf_var.global_modulation_freqs, conf_var.global_tdir_modulations,
                                conf_var.global_clock_asds, conf_var.global_clock_offsets,
                                conf_var.global_clock_freqoffsets, conf_var.global_clock_freqlindrifts, \
                                conf_var.global_clock_freqquaddrifts, conf_var.global_clockinv_tolerance,
                                conf_var.global_clockinv_maxiter,
                                conf_var.global_backlink_asds, conf_var.global_global_backlink_fknees,
                                conf_var.global_testmass_asds, \
                                conf_var.global_testmass_fknees, conf_var.global_testmass_fbreak,
                                conf_var.global_testmass_shape,
                                conf_var.global_testmass_frelax, conf_var.global_oms_asds, conf_var.global_oms_fknees, \
                                conf_var.global_moc_time_correlation_asds, conf_var.global_ttl_coeffs,
                                conf_var.global_sc_jitter_asds,
                                conf_var.global_sc_jitter_fknees, conf_var.global_mosa_jitter_asds,
                                conf_var.global_mosa_jitter_fknees, \
                                conf_var.global_mosa_angles, conf_var.global_dws_asds, conf_var.global_ranging_biases,
                                conf_var.global_ranging_asds,
                                conf_var.global_prn_ambiguity, conf_var.global_electro_delays,
                                conf_var.global_concurrent)

    # instrument.disable_all_noises(but='clock')
    # instrument.disable_dopplers()

    # Write to a measurement file
    instrument.simulate(keep_all=True)
    instrument.write('all-measurements-new-4.h5', keep_all=True)

    # ## plot to pdf
    # instrument.plot_offsets(output='ISI_RFI_TMI_carrier_offsets.pdf', skip=500)
    # instrument.plot_fluctuations(output='ISI_RFI_TMI_carrier_fluctuations.pdf', skip=500)
    # instrument.plot_totals(output='ISI_RFI_TMI_carrier.pdf', skip=500)
    # instrument.plot_mprs(output='measured_mpprs_dmpprs.pdf', skip=500)
    # instrument.plot_dws(output='isi_dws_phis_etas.pdf', skip=500)


def my_test():
    version = importlib_metadata.version('lisainstrument')
    print_ver = "lisainstrument version: " + str(version)
    print(print_ver)


if __name__ == '__main__':
    # get_config()
    my_test()
    main(0)
