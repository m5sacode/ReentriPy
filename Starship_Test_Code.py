import reentripy as rpy
import numpy as np
from pystdatm import density

# ------------------------------
# Starship parameters (empty)
# ------------------------------
cl = 1.2
cd = 1.3
area = 545.0       # m^2
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

# ------------------------------
# Orbit definition: Conditions for IFT test flights ( more or less )
# ------------------------------
apogee = 213_000.0
perigee = -15_000.0
altitude = 200_000.0   # current altitude

inclination = np.deg2rad(26.8)     # Starship-like
arg_perigee = np.deg2rad(-35.0)
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
sc.run_reentry(gif=False, controller=False) # Max lift reentry


sc.keplerian_initial_conditions(
    apogee=apogee,
    perigee=perigee,
    altitude=altitude,
    inclination=inclination,
    arg_perigee=arg_perigee,
    raan=raan,
    true_anomaly_sign=-1  # descending branch (reentry)
)


sc.run_reentry(gif=False, controller="PDR") # With controller adjusting bank trying to keep DR to 0

sc.keplerian_initial_conditions(
    apogee=apogee,
    perigee=perigee,
    altitude=altitude,
    inclination=inclination,
    arg_perigee=arg_perigee,
    raan=raan,
    true_anomaly_sign=-1  # descending branch (reentry)
)


sc.run_reentry(gif=False, controller="PH") # With controller adjusting bank trying to keep optimum altitude

sc.keplerian_initial_conditions(
    apogee=apogee,
    perigee=perigee,
    altitude=altitude,
    inclination=inclination,
    arg_perigee=arg_perigee,
    raan=raan,
    true_anomaly_sign=-1  # descending branch (reentry)
)


sc.run_reentry(gif=False, controller="PQC") # With controller adjusting bank trying to keep max heating (DOESN'T REALLY WORK --> HIGH Gs)
#
#
#
# sc.keplerian_initial_conditions(
#     apogee=apogee,
#     perigee=perigee,
#     altitude=altitude,
#     inclination=inclination,
#     arg_perigee=arg_perigee,
#     raan=raan,
#     true_anomaly_sign=-1  # descending branch (reentry)
# )
#
# sc.banking_angle = np.deg2rad(90)
# sc.run_reentry(gif=False, controller=False) # Max bank

# banks = [90, 80, 70, 60, 50, 40, 30, 20, 0]
# for bank in banks:
#     sc.keplerian_initial_conditions(
#         apogee=apogee,
#         perigee=perigee,
#         altitude=altitude,
#         inclination=inclination,
#         arg_perigee=arg_perigee,
#         raan=raan,
#         true_anomaly_sign=-1  # descending branch (reentry)
#     )
#     sc.banking_angle = np.deg2rad(bank)
#     sc.run_reentry(gif=False, controller=False)