import reentripy as rpy
import numpy as np
from pystdatm import density
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D

# ------------------------------
# Starship parameters (empty)
# ------------------------------
cl = 1.2
cd = 1.3
area = 63.82       # m^2
mass = 120_000.0    # kg

# Peak heating: IFT 11 T+ 51' 51'' --> 24783 kmh 70.2 km

peak_heating_speed = 24783/3.6
peak_heating_alt = 70200.0

peak_heating_rho = density(peak_heating_alt)

nose_radius = 3

k = 1.7415e-4  # (Earth)
# k = 1.9027e-4  # (Mars)
qc_max = k * np.sqrt(peak_heating_rho/nose_radius) * peak_heating_speed ** 3



# Create spacecraft
sc = rpy.Spacecraft(cl=cl, cd=cd, A=area, m=mass, max_qc=qc_max, nose_radius=nose_radius)

sc.load_aero_tables(
    "Starship Aero Data/wpd_starship_cl.csv",
    "Starship Aero Data/wpd_starship_cd.csv"
)
# print(sc.mach_max)
# print(sc.mach_min)
#
# sc.plot_aero_interpolation(n_mach=100, n_aoa=100, mach_min=sc.mach_min, mach_max=sc.mach_max, aoa_max=sc.aoa_max, aoa_min=sc.aoa_min)


# ------------------------------
# Orbit definition: Conditions for IFT test flights ( more or less )
# ------------------------------
apogee = 213_000.0
perigee = -15_000.0
altitude = 200_000.0   # current altitude

inclination = np.deg2rad(26.8)     # Starship-like
arg_perigee = np.deg2rad(-5.0)
raan = np.deg2rad(180.0)

# ------------------------------
# Generate Cartesian state
# ------------------------------
sc.keplerian_initial_conditions(
    apogee=apogee,
    perigee=perigee,
    altitude=altitude,
    inclination=inclination,
    arg_perigee=arg_perigee,
    raan=raan,
    true_anomaly_sign=-1  # descending branch (reentry)
)

print("Position vector (m):")
print(sc.position_vector)

print("\nVelocity vector (m/s):")
print(sc.cart_velocity_vector)

print("\nSpeed (m/s):")
print(np.linalg.norm(sc.cart_velocity_vector))

# ------------------------------
# Plot orbit & spacecraft
# ------------------------------
# sc.plot_orbit_3d_init(
#     apogee=apogee,
#     perigee=perigee,
#     inclination=inclination,
#     arg_perigee=arg_perigee,
#     raan=raan)
# sc.plot_orbit_3d()


sc.banking_angle = 0

results = {}
results["Max Lift"] = sc.run_reentry(gif=False, controller=False) # Max lift reentry


sc.keplerian_initial_conditions(
    apogee=apogee,
    perigee=perigee,
    altitude=altitude,
    inclination=inclination,
    arg_perigee=arg_perigee,
    raan=raan,
    true_anomaly_sign=-1  # descending branch (reentry)
)


results["Min Descend Rate"] = sc.run_reentry(gif=False, controller="PDR") # With controller adjusting bank trying to keep DR to 0

sc.keplerian_initial_conditions(
    apogee=apogee,
    perigee=perigee,
    altitude=altitude,
    inclination=inclination,
    arg_perigee=arg_perigee,
    raan=raan,
    true_anomaly_sign=-1  # descending branch (reentry)
)


# results["Smart Min DR Altitude Control"] = sc.run_reentry(gif=False, controller="PH") # With controller adjusting bank trying to keep optimum altitude

sc.keplerian_initial_conditions(
    apogee=apogee,
    perigee=perigee,
    altitude=altitude,
    inclination=inclination,
    arg_perigee=arg_perigee,
    raan=raan,
    true_anomaly_sign=-1  # descending branch (reentry)
)


results["Heating Peak Widening Control"] = sc.run_reentry(gif=False, controller="PQC") # With controller adjusting bank trying to keep max heating (DOESN'T REALLY WORK --> HIGH Gs)

sc.keplerian_initial_conditions(
    apogee=apogee,
    perigee=perigee,
    altitude=altitude,
    inclination=inclination,
    arg_perigee=arg_perigee,
    raan=raan,
    true_anomaly_sign=-1  # descending branch (reentry)
)


results["Min Descend Rate (alpha)"] = sc.run_reentry(gif=False, controller="aPDR") # With controller adjusting bank trying to keep DR to 0

sc.keplerian_initial_conditions(
    apogee=apogee,
    perigee=perigee,
    altitude=altitude,
    inclination=inclination,
    arg_perigee=arg_perigee,
    raan=raan,
    true_anomaly_sign=-1  # descending branch (reentry)
)


results["Heating Peak Widening Control (alpha)"] = sc.run_reentry(gif=False, controller="aPQC") # With controller adjusting bank trying to keep max heating (DOESN'T REALLY WORK --> HIGH Gs)


colors = {
    "Max Lift": "#1f77b4",                      # blue
    "Min Descend Rate": "#ff7f0e",              # orange
    # "Smart Min DR Altitude Control": "#2ca02c",  # green
    "Heating Peak Widening Control": "#d62728", # red

    # Lighter variants
    "Min Descend Rate (alpha)": "#ffbb78",      # light orange
    "Heating Peak Widening Control (alpha)": "#ff9896",  # light red
}


alpha = 0.85

fig = plt.figure(figsize=(20, 28))
gs = gridspec.GridSpec(6, 2, figure=fig, height_ratios=[1,1,1,1,1,3])

ax_alt = fig.add_subplot(gs[0,0])
ax_speed = fig.add_subplot(gs[0,1])
ax_mach = fig.add_subplot(gs[1,0])
ax_speed_alt = fig.add_subplot(gs[1,1])
ax_mach_alt = fig.add_subplot(gs[2,0])
ax_bank = fig.add_subplot(gs[2,1])
ax_g = fig.add_subplot(gs[3,0])
ax_descent = fig.add_subplot(gs[3,1])
ax_qdyn = fig.add_subplot(gs[4,0])
ax_heat = fig.add_subplot(gs[4,1])
ax_heat_load = ax_heat.twinx()
ax_gt = fig.add_subplot(gs[5,:], projection=ccrs.PlateCarree())
ax_gt.stock_img()
ax_gt.add_feature(cfeature.LAND, facecolor="lightgray")
ax_gt.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax_gt.add_feature(cfeature.COASTLINE)
ax_gt.set_title("Ground Track Comparison")
ax_gt.set_xlabel("Longitude (deg)")
ax_gt.set_ylabel("Latitude (deg)")

for name, data in results.items():
    (times, alt, speed, mach, bank,
     g, dr, pos, lon, lat,
     heat_load, heat_flux, aoas) = data

    c = colors[name]

    ax_alt.plot(times, alt, color=c, alpha=alpha, label=name)
    ax_speed.plot(times, speed, color=c, alpha=alpha)
    ax_mach.plot(times, mach, color=c, alpha=alpha)

    ax_speed_alt.plot(alt, speed, color=c, alpha=alpha)
    ax_mach_alt.plot(alt, mach, color=c, alpha=alpha)

    ax_bank.plot(times, bank, color=c, alpha=alpha)
    ax_g.plot(times, g, color=c, alpha=alpha)
    ax_descent.plot(times, dr, color=c, alpha=alpha)
    ax_heat.plot(
        times,
        heat_flux / 1e4,
        color=c,
        alpha=alpha
    )

    ax_heat_load.plot(
        times,
        heat_load / 1e7,
        color=c,
        alpha=alpha,
        linestyle="--"
    )
    lon = np.rad2deg(data[8])
    lat = np.rad2deg(data[9])

    ax_gt.plot(
        lon,
        lat,
        color=colors[name],
        alpha=alpha,
        linewidth=2,
        label=name,
        transform=ccrs.Geodetic()
    )



legend_handles = [
    Line2D(
        [0], [0],
        color=colors[name],
        lw=3,
        label=name
    )
    for name in results.keys()
]


ax_alt.set_title("Altitude vs Time")
ax_alt.set_xlabel("Time (s)")
ax_alt.set_ylabel("Altitude (m)")
ax_alt.legend()

ax_speed.set_title("Speed vs Time")
ax_speed.set_xlabel("Time (s)")
ax_speed.set_ylabel("Speed (m/s)")

ax_mach.set_title("Mach vs Time")
ax_mach.set_xlabel("Time (s)")
ax_mach.set_ylabel("Mach")

ax_speed_alt.set_title("Speed vs Altitude")
ax_speed_alt.set_xlabel("Altitude (m)")
ax_speed_alt.set_ylabel("Speed (m/s)")

ax_mach_alt.set_title("Mach vs Altitude")
ax_mach_alt.set_xlabel("Altitude (m)")
ax_mach_alt.set_ylabel("Mach")

ax_bank.set_title("Bank Angle vs Time")
ax_bank.set_xlabel("Time (s)")
ax_bank.set_ylabel("Bank (deg)")

ax_g.set_title("g-load vs Time")
ax_g.set_xlabel("Time (s)")
ax_g.set_ylabel("g")

ax_descent.set_title("Descent Rate vs Time")
ax_descent.set_xlabel("Time (s)")
ax_descent.set_ylabel("m/s")

ax_heat.set_title("Heat Flux & Integrated Heat Load vs Time")
ax_heat.set_xlabel("Time (s)")
ax_heat.set_ylabel("Heat Flux (×1e4 W/m²)", color="red")
ax_heat_load.set_ylabel("Heat Load (×1e7 J/m²)", color="black")
ax_gt.legend(loc="lower left")


global_max_qdot = max(
    np.max(data[-1]) for data in results.values()
) / 1e4

ax_heat.axhline(
    global_max_qdot,
    color="red",
    linestyle=":",
    linewidth=2,
    alpha=0.8
)


for ax in [ax_alt, ax_speed, ax_mach, ax_speed_alt,
           ax_mach_alt, ax_bank, ax_g, ax_descent,
           ax_heat]:
    ax.grid(True, alpha=0.3)


ax_qdyn.legend(
    handles=legend_handles,
    loc="center",
    frameon=True,
    title="Controller",
    fontsize=11,
    title_fontsize=12
)

ax_qdyn.set_xticks([])
ax_qdyn.set_yticks([])
ax_qdyn.set_title("Controller Legend")


plt.tight_layout()
plt.show()
