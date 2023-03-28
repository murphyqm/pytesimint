# Preliminary Reference Asteroidal Model (PRAM)

# calculating the gravity at depth within a body
import numpy as np
import matplotlib.pyplot as plt


def volume_layers(r_inner, r_outer):
    v_inner = (4.0/3.0) * np.pi * (r_inner**3)
    v_outer = (4.0/3.0) * np.pi * ((r_outer**3) - (r_inner**3))
    return v_inner, v_outer


def vol_sphere(r):
    vol = (4.0/3.0) * np.pi * (r**3)
    return vol


def radius_from_vol(vol):
    r = np.cbrt((3/4) * (vol/(np.pi)))
    return r


def mass(volume, density):
    m = volume * density
    return m


def avg_density(vol_frac1,
                density_1,
                density_2):
    inv_rho_mix = (vol_frac1/density_1) + ((1.0-vol_frac1)/density_2)
    rho_mix = 1.0/inv_rho_mix
    return rho_mix


def total_mass_at_r(r_inner, r_outer, density_inner, density_outer):
    v_inner, v_outer = volume_layers(r_inner, r_outer)
    m_inner = mass(v_inner, density_inner)
    m_outer = mass(v_outer, density_outer)
    total_mass = m_inner + m_outer
    return total_mass


def gravity_at_r(r, mass_at_r):
    G = 6.67408E-11
    g = (G * mass_at_r)/(r**2)
    return g


def gravity_in_space(total_mass, r):
    G = 6.67408E-11
    g = (G * total_mass)/(r**2)
    return g


def grav_in_core(r, density):
    volume = vol_sphere(r)
    mass_core = mass(volume, density)
    grav = gravity_at_r(r, mass_core)
    return grav


def stokes_velocity(g, dens_contrast, diameter, dynamic_visc):
    vel = (g * dens_contrast * (diameter**2))/(18.0 * dynamic_visc)
    return vel


def settling_time(displacement, velocity):
    time = displacement/velocity
    return time


def all_in_one(total_radius_km,
               core_radius_km,
               density_mantle,
               density_core,
               particle_diam,
               dynamic_visc,
               settling_disp_m):
    total_radius = total_radius_km * 1000.0
    core_radius = core_radius_km * 1000.0
    radii = np.arange(core_radius+1, total_radius+1)
    radii_in_km = radii/1000
    dens_contrast = density_core - density_mantle
    mass_list = total_mass_at_r(core_radius, radii, density_core, density_mantle)
    gravity_list = gravity_at_r(radii, mass_list)
    radii_core = np.linspace(1000, core_radius+1)
    gravity_core = grav_in_core(radii_core, density_core)
    core_radii_km = radii_core/1000
    velocities = stokes_velocity(gravity_list,
                                 dens_contrast,
                                 particle_diam,
                                 dynamic_visc)
    settling_times = settling_time(settling_disp_m, velocities)
    settling_times_mins = settling_times/60.0

    # plotting g and stokes vel/settling time together
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax1 = plt.subplots()

    ax1.set_xlim(0,total_radius/1000)
    ax1.axvspan(0, core_radius/1000, alpha=0.2, color="grey")

    ax1.plot(radii_in_km, gravity_list, c="k")
    ax1.plot(core_radii_km, gravity_core, c="k")

    ax1.axvline(x=core_radius/1000,
                color="grey",
                ls="dotted",
                label="Core-mantle boundary")
    ax1.set_xlabel("Radius [km]")
    ax1.set_ylabel("Acceleration due to gravity [m s$^{-2}$]")

    # ax1.legend(frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(radii_in_km, settling_times_mins, ls="dashed")
    ax2.tick_params(axis="y", labelcolor=cycle_colors[0])
    ax2.set_ylabel(f"Settling time for {int(settling_disp_m)} m displacement [min]",
                   color=cycle_colors[0])

    mantle_text_loc = (core_radius + ((total_radius - core_radius)/2))/1000
    ax1.text(core_radius/2000.0, 0.0, "Core", ha='center',)  # va='center')
    ax1.text(mantle_text_loc, 0.0, "Mantle", ha='center',)  # va='center')


def all_in_one_axes(total_radius_km,
                    core_radius_km,
                    density_mantle,
                    density_core,
                    particle_diam,
                    dynamic_visc,
                    settling_disp_m,
                    fig,
                    ax1,
                    ax_label=""):
    total_radius = total_radius_km * 1000.0
    core_radius = core_radius_km * 1000.0
    radii = np.arange(core_radius+1, total_radius+1)
    radii_in_km = radii/1000
    dens_contrast = density_core - density_mantle
    mass_list = total_mass_at_r(core_radius, radii, density_core, density_mantle)
    gravity_list = gravity_at_r(radii, mass_list)
    radii_core = np.linspace(1000, core_radius+1)
    gravity_core = grav_in_core(radii_core, density_core)
    core_radii_km = radii_core/1000
    velocities = stokes_velocity(gravity_list,
                                 dens_contrast,
                                 particle_diam,
                                 dynamic_visc)
    settling_times = settling_time(settling_disp_m, velocities)
    settling_times_mins = settling_times/60.0

    # plotting g and stokes vel/settling time together
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # fig, ax1 = plt.subplots()

    ax1.set_xlim(0, total_radius/1000)
    ax1.axvspan(0, core_radius/1000, alpha=0.2, color="grey")

    ax1.plot(radii_in_km, gravity_list, c="k")
    ax1.plot(core_radii_km, gravity_core, c="k")

    # ax1.axvline(x=core_radius/1000,
    #             color="grey",
    #             ls="dotted",
    #             label="Core-mantle boundary")
    ax1.set_xlabel("Radius [km]")
    ax1.set_ylabel("g [m s$^{-2}$]")

    # ax1.legend(frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(radii_in_km, settling_times_mins, ls="dashed")
    ax2.tick_params(axis="y", labelcolor=cycle_colors[0])
    ax2.set_ylabel(f"Settling time [min]\ns={int(settling_disp_m)} m, d={particle_diam} m",
                   color=cycle_colors[0])

    mantle_text_loc = (core_radius + ((total_radius - core_radius)/2))/1000
    ax1.text(core_radius/2000.0, 0.0, "Core", ha='center',)  # va='center')
    ax1.text(mantle_text_loc, 0.0, "Mantle", ha='center',)  # va='center')
    ax1.text(0.02, 0.9, ax_label,
             horizontalalignment='left',
             verticalalignment='center',
             transform=ax1.transAxes)
    return ax1, ax2


def all_in_one_axes_diff_labels(total_radius_km,
                                core_radius_km,
                                density_mantle,
                                density_core,
                                particle_diam,
                                dynamic_visc,
                                settling_disp_m,
                                fig,
                                ax1,
                                ax_label="",
                                lab_x=-0.115,
                                lab_y=0.9):
    total_radius = total_radius_km * 1000.0
    core_radius = core_radius_km * 1000.0
    radii = np.arange(core_radius+1, total_radius+1)
    radii_in_km = radii/1000
    dens_contrast = density_core - density_mantle
    mass_list = total_mass_at_r(core_radius, radii, density_core, density_mantle)
    gravity_list = gravity_at_r(radii, mass_list)
    radii_core = np.linspace(1000, core_radius+1)
    gravity_core = grav_in_core(radii_core, density_core)
    core_radii_km = radii_core/1000
    velocities = stokes_velocity(gravity_list,
                                 dens_contrast,
                                 particle_diam,
                                 dynamic_visc)
    settling_times = settling_time(settling_disp_m, velocities)
    settling_times_mins = settling_times/60.0

    # plotting g and stokes vel/settling time together
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # fig, ax1 = plt.subplots()

    ax1.set_xlim(0, total_radius/1000)
    ax1.axvspan(0, core_radius/1000, alpha=0.2, color="grey")

    ax1.plot(radii_in_km, gravity_list, c="k")
    ax1.plot(core_radii_km, gravity_core, c="k")

    # ax1.axvline(x=core_radius/1000,
    #             color="grey",
    #             ls="dotted",
    #             label="Core-mantle boundary")
    ax1.set_xlabel("Radius [km]")
    ax1.set_ylabel("g [m s$^{-2}$]")

    # ax1.legend(frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(radii_in_km, settling_times_mins, ls="dashed", label = f"d = {particle_diam} m")
    ax2.tick_params(axis="y", labelcolor=cycle_colors[0])
    ax2.set_ylabel(f"Settling time [min]\ns={int(settling_disp_m)} m",
                   color=cycle_colors[0])

    mantle_text_loc = (core_radius + ((total_radius - core_radius)/2))/1000
    ax1.text(core_radius/2000.0, 0.0, "Core", ha='center',)  # va='center')
    ax1.text(mantle_text_loc, 0.0, "Mantle", ha='center',)  # va='center')
    ax1.text(lab_x, lab_y, ax_label,
             horizontalalignment='left',
             verticalalignment='center',
             transform=ax1.transAxes)
    return ax1, ax2


def all_in_one_axes_three_viscs(total_radius_km,
                                core_radius_km,
                                density_mantle,
                                density_core,
                                particle_diam,
                                dynamic_visc1,
                                dynamic_visc2,
                                dynamic_visc3,
                                settling_disp_m,
                                fig,
                                ax1,
                                ax_label="",
                                lab_x=-0.115,
                                lab_y=0.9):
    total_radius = total_radius_km * 1000.0
    core_radius = core_radius_km * 1000.0
    radii = np.arange(core_radius+1, total_radius+1)
    radii_in_km = radii/1000
    dens_contrast = density_core - density_mantle
    mass_list = total_mass_at_r(core_radius, radii, density_core, density_mantle)
    gravity_list = gravity_at_r(radii, mass_list)
    radii_core = np.linspace(1000, core_radius+1)
    gravity_core = grav_in_core(radii_core, density_core)
    core_radii_km = radii_core/1000
    velocities1 = stokes_velocity(gravity_list,
                                  dens_contrast,
                                  particle_diam,
                                  dynamic_visc1)
    settling_times1 = settling_time(settling_disp_m, velocities1)
    settling_times_mins1 = settling_times1/60.0

    velocities2 = stokes_velocity(gravity_list,
                                  dens_contrast,
                                  particle_diam,
                                  dynamic_visc2)
    settling_times2 = settling_time(settling_disp_m, velocities2)
    settling_times_mins2 = settling_times2/60.0

    velocities3 = stokes_velocity(gravity_list,
                                  dens_contrast,
                                  particle_diam,
                                  dynamic_visc3)
    settling_times3 = settling_time(settling_disp_m, velocities3)
    settling_times_mins3 = settling_times3/60.0

    # plotting g and stokes vel/settling time together
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # fig, ax1 = plt.subplots()

    ax1.set_xlim(0, total_radius/1000)
    ax1.axvspan(0, core_radius/1000, alpha=0.2, color="grey")

    ax1.plot(radii_in_km, gravity_list, c="k")
    ax1.plot(core_radii_km, gravity_core, c="k")

    # ax1.axvline(x=core_radius/1000,
    #             color="grey",
    #             ls="dotted",
    #             label="Core-mantle boundary")
    ax1.set_xlabel("Radius [km]")
    ax1.set_ylabel("g [m s$^{-2}$]")

    # ax1.legend(frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(radii_in_km, settling_times_mins1, ls="dashed", label=f"$\mu$ = {dynamic_visc1} Pa s")
    ax2.plot(radii_in_km, settling_times_mins2, ls="dashed", label = f"$\mu$ = {dynamic_visc2} Pa s")
    ax2.plot(radii_in_km, settling_times_mins3, ls="dashed", label = f"$\mu$ = {dynamic_visc3} Pa s")
    ax2.tick_params(axis="y", labelcolor=cycle_colors[0])
    ax2.set_ylabel(f"Settling time [min]\ns={int(settling_disp_m)} m",
                   color=cycle_colors[0])

    mantle_text_loc = (core_radius + ((total_radius - core_radius)/2))/1000
    ax1.text(core_radius/2000.0, 0.0, "Core", ha='center',)  # va='center')
    ax1.text(mantle_text_loc, 0.0, "Mantle", ha='center',)  # va='center')
    ax1.text(lab_x, lab_y, ax_label,
             horizontalalignment='left',
             verticalalignment='center',
             transform=ax1.transAxes)
    return ax1, ax2


def all_in_one_axes_three_diams(total_radius_km,
                                core_radius_km,
                                density_mantle,
                                density_core,
                                particle_diam1,
                                particle_diam2,
                                particle_diam3,
                                dynamic_visc,
                                settling_disp_m,
                                fig,
                                ax1,
                                ax_label="",
                                lab_x=-0.115,
                                lab_y=0.9):
    total_radius = total_radius_km * 1000.0
    core_radius = core_radius_km * 1000.0
    radii = np.arange(core_radius+1, total_radius+1)
    radii_in_km = radii/1000
    dens_contrast = density_core - density_mantle
    mass_list = total_mass_at_r(core_radius, radii, density_core, density_mantle)
    gravity_list = gravity_at_r(radii, mass_list)
    radii_core = np.linspace(1000, core_radius+1)
    gravity_core = grav_in_core(radii_core, density_core)
    core_radii_km = radii_core/1000
    velocities1 = stokes_velocity(gravity_list,
                                  dens_contrast,
                                  particle_diam1,
                                  dynamic_visc)
    settling_times1 = settling_time(settling_disp_m, velocities1)
    settling_times_mins1 = settling_times1/60.0

    velocities2 = stokes_velocity(gravity_list,
                                  dens_contrast,
                                  particle_diam2,
                                  dynamic_visc)
    settling_times2 = settling_time(settling_disp_m, velocities2)
    settling_times_mins2 = settling_times2/60.0

    velocities3 = stokes_velocity(gravity_list,
                                  dens_contrast,
                                  particle_diam3,
                                  dynamic_visc)
    settling_times3 = settling_time(settling_disp_m, velocities3)
    settling_times_mins3 = settling_times3/60.0

    # plotting g and stokes vel/settling time together
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # fig, ax1 = plt.subplots()

    ax1.set_xlim(0, total_radius/1000)
    ax1.axvspan(0, core_radius/1000, alpha=0.2, color="grey")

    ax1.plot(radii_in_km, gravity_list, c="k")
    ax1.plot(core_radii_km, gravity_core, c="k")

    # ax1.axvline(x=core_radius/1000,
    #             color="grey",
    #             ls="dotted",
    #             label="Core-mantle boundary")
    ax1.set_xlabel("Radius [km]")
    ax1.set_ylabel("g [m s$^{-2}$]")

    # ax1.legend(frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(radii_in_km, settling_times_mins1, ls="dashed", label=f"d = {particle_diam1} m")
    ax2.plot(radii_in_km, settling_times_mins2, ls="dashed", label = f"d = {particle_diam2} m")
    ax2.plot(radii_in_km, settling_times_mins3, ls="dashed", label = f"d = {particle_diam3} m")
    ax2.tick_params(axis="y", labelcolor=cycle_colors[0])
    ax2.set_ylabel(f"Settling time [min]\ns={int(settling_disp_m)} m",
                   color=cycle_colors[0])
    # ax2.set_yscale("log")

    mantle_text_loc = (core_radius + ((total_radius - core_radius)/2))/1000
    ax1.text(core_radius/2000.0, 0.0, "Core", ha='center',)  # va='center')
    ax1.text(mantle_text_loc, 0.0, "Mantle", ha='center',
             bbox=dict(facecolor='white', edgecolor='white'))  # va='center')
    ax1.text(lab_x, lab_y, ax_label,
             horizontalalignment='left',
             verticalalignment='center',
             transform=ax1.transAxes)
    return ax1, ax2


def all_in_one_axes_two_diams(total_radius_km,
                              core_radius_km,
                              density_mantle,
                              density_core,
                              particle_diam1,
                              particle_diam2,
                              dynamic_visc,
                              settling_disp_m,
                              fig,
                              ax1,
                              ax_label="",
                              lab_x=-0.112,
                              lab_y=0.9,
                              core_lab="Core",
                              mantle_lab="Mantle",
                              text_y_loc=145):
    total_radius = total_radius_km * 1000.0
    core_radius = core_radius_km * 1000.0
    radii = np.arange(core_radius+1, total_radius+1)
    radii_space = np.arange(total_radius+1, total_radius+50)
    radii_in_km = radii/1000
    radii_space_km = radii_space/1000
    dens_contrast = density_core - density_mantle
    mass_list = total_mass_at_r(core_radius, radii, density_core, density_mantle)
    gravity_list = gravity_at_r(radii, mass_list)
    gravity_space = gravity_in_space(mass_list[-1], radii_space)
    radii_core = np.linspace(1000, core_radius+1)
    gravity_core = grav_in_core(radii_core, density_core)
    core_radii_km = radii_core/1000
    velocities1 = stokes_velocity(gravity_list,
                                  dens_contrast,
                                  particle_diam1,
                                  dynamic_visc)
    settling_times1 = settling_time(settling_disp_m, velocities1)
    settling_times_mins1 = settling_times1/60.0

    velocities2 = stokes_velocity(gravity_list,
                                  dens_contrast,
                                  particle_diam2,
                                  dynamic_visc)
    settling_times2 = settling_time(settling_disp_m, velocities2)
    settling_times_mins2 = settling_times2/60.0

    # plotting g and stokes vel/settling time together
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # fig, ax1 = plt.subplots()

    ax1.set_xlim(0, total_radius/1000)
    ax1.axvspan(0, core_radius/1000, alpha=0.2, color="grey")

    ax1.plot(radii_in_km, gravity_list, c="k")
    ax1.plot(core_radii_km, gravity_core, c="k")
    # ax1.plot(radii_space_km, gravity_space, c="k")

    ax1.axvline(x=core_radius/1000,
                color="grey",
                alpha=0.4,
                label="Core-mantle boundary")
    ax1.set_xlabel("Radius [km]")
    ax1.set_ylabel("g [m s$^{-2}$]")

    # ax1.legend(frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(radii_in_km, settling_times_mins1, ls="dashed",
             c=cycle_colors[0], label=f"d = {particle_diam1} m",
             zorder=2)
    ax2.plot(radii_in_km, settling_times_mins2, ls="dotted",
             c=cycle_colors[0], label=f"d = {particle_diam2} m",
             lw=2, zorder=2)
    ax2.tick_params(axis="y", labelcolor=cycle_colors[0])
    ax2.set_ylabel(f"Settling time [min]\ns={int(settling_disp_m)} m",
                   color=cycle_colors[0])
    # ax2.set_yscale("log")

    mantle_text_loc = (core_radius + ((total_radius - core_radius)/2))/1000
    ax2.text(core_radius/2000.0, text_y_loc, core_lab, ha='center',)
             # bbox=dict(facecolor='white', edgecolor='grey',
             #           boxstyle="round",))  # va='center')
    ax2.text(mantle_text_loc, text_y_loc, mantle_lab, ha='center',
             zorder=3,)
             # bbox=dict(facecolor='white', edgecolor='grey',
             #           boxstyle="round",))  # va='center')
    ax1.text(lab_x, lab_y, ax_label,
             horizontalalignment='left',
             verticalalignment='center',
             transform=ax1.transAxes)
    return ax1, ax2


def all_in_one_axes_two_diams2(total_radius_km,
                               core_radius_km,
                               density_mantle,
                               density_core,
                               particle_diam1,
                               particle_diam2,
                               dynamic_visc,
                               settling_disp_m,
                               fig,
                               ax1,
                               ax_label="",
                               lab_x=-0.112,
                               lab_y=0.9,
                               core_lab="Core",
                               mantle_lab="Mantle",
                               text_y_loc=145):
    total_radius = total_radius_km * 1000.0
    core_radius = core_radius_km * 1000.0
    radii = np.arange(core_radius+1, total_radius+1)
    radii_in_km = radii/1000
    dens_contrast = density_core - density_mantle
    mass_list = total_mass_at_r(core_radius, radii, density_core, density_mantle)
    gravity_list = gravity_at_r(radii, mass_list)
    radii_core = np.linspace(1000, core_radius+1)
    gravity_core = grav_in_core(radii_core, density_core)
    core_radii_km = radii_core/1000
    radii_space = np.arange(total_radius+1, total_radius+50000)
    radii_space_km = radii_space/1000
    gravity_space = gravity_in_space(mass_list[-1], radii_space)
    velocities1 = stokes_velocity(gravity_list,
                                  dens_contrast,
                                  particle_diam1,
                                  dynamic_visc)
    settling_times1 = settling_time(settling_disp_m, velocities1)
    settling_times_mins1 = settling_times1/60.0

    velocities2 = stokes_velocity(gravity_list,
                                  dens_contrast,
                                  particle_diam2,
                                  dynamic_visc)
    settling_times2 = settling_time(settling_disp_m, velocities2)
    settling_times_mins2 = settling_times2/60.0

    # plotting g and stokes vel/settling time together
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # fig, ax1 = plt.subplots()

    # ax1.set_xlim(0, total_radius/1000)
    ax1.axvline(x=total_radius/1000,
                color="k",
                alpha=0.4,
                label="Radius")
    ax1.axvspan(0, core_radius/1000, alpha=0.2, color="grey")

    ax1.plot(radii_in_km, gravity_list, c="k")
    ax1.plot(core_radii_km, gravity_core, c="k")
    ax1.plot(radii_space_km, gravity_space, c="k",)
    # print(radii_space_km)

    ax1.axvline(x=core_radius/1000,
                color="grey",
                alpha=0.4,
                label="Core-mantle boundary")
    ax1.set_xlabel("Radius [km]")
    ax1.set_ylabel("g [m s$^{-2}$]")

    # ax1.legend(frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(radii_in_km, settling_times_mins1, ls="dashed",
             c=cycle_colors[0], label=f"d = {particle_diam1} m",
             zorder=2)
    ax2.plot(radii_in_km, settling_times_mins2, ls="dotted",
             c=cycle_colors[0], label=f"d = {particle_diam2} m",
             lw=2, zorder=2)
    ax2.tick_params(axis="y", labelcolor=cycle_colors[0])
    ax2.set_ylabel(f"Settling time [min]\ns={int(settling_disp_m)} m",
                   color=cycle_colors[0])
    # ax2.set_yscale("log")

    mantle_text_loc = (core_radius + ((total_radius - core_radius)/2))/1000
    ax2.text(core_radius/2000.0, text_y_loc, core_lab, ha='center',)
             # bbox=dict(facecolor='white', edgecolor='grey',
             #           boxstyle="round",))  # va='center')
    ax2.text(mantle_text_loc, text_y_loc, mantle_lab, ha='center',
             zorder=3,)
             # bbox=dict(facecolor='white', edgecolor='grey',
             #           boxstyle="round",))  # va='center')
    ax1.text(lab_x, lab_y, ax_label,
             horizontalalignment='left',
             verticalalignment='center',
             transform=ax1.transAxes)
    return ax1, ax2