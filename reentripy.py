import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from pystdatm import density, temperature, speed_of_sound
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.optimize import brentq


def eci_to_lonlat(r_vec_eci, t, planet_radius=6_371_000.0):
    """
    Converts ECI Cartesian position vector(s) to geodetic latitude and longitude (radians),
    accounting for Earth's rotation at time t (s since epoch).

    Parameters
    ----------
    r_vec_eci : ndarray
        Nx3 array of ECI positions (m)
    t : float or ndarray
        Time(s) since reference epoch (s)
    planet_radius : float
        Radius of planet (m)

    Returns
    -------
    lon, lat : ndarray
        Longitude and latitude in radians
    """
    r_vec_eci = np.atleast_2d(r_vec_eci)

    # Earth's rotation rate (rad/s)
    omega = 2 * np.pi / 86164.0

    # Rotation angle
    theta = omega * t  # rotate from ECI to ECEF

    # Handle vectorized time
    if np.ndim(t) > 0 and len(t) == r_vec_eci.shape[0]:
        r_ecef = np.zeros_like(r_vec_eci)
        for i, ti in enumerate(t):
            c, s = np.cos(-omega * ti), np.sin(-omega * ti)
            Rz = np.array([[c, -s, 0],
                           [s, c, 0],
                           [0, 0, 1]])
            r_ecef[i] = Rz @ r_vec_eci[i]
    else:
        # Single time for all positions
        c, s = np.cos(-theta), np.sin(-theta)
        Rz = np.array([[c, -s, 0],
                       [s, c, 0],
                       [0, 0, 1]])
        r_ecef = (Rz @ r_vec_eci.T).T

    x, y, z = r_ecef.T
    lon = np.arctan2(y, x)
    lat = np.arcsin(z / np.linalg.norm(r_ecef, axis=1))

    return lon, lat


def altitude_for_density(rho_target, h_min=0, h_max=79_000):
    """
    Find altitude (m) where atmospheric density equals rho_target (kg/m^3)
    using pystdatm.
    h_min, h_max : search range in meters
    """
    # Define function: f(h) = density(h) - rho_target
    f = lambda h: atmospheric_properties(h)[0] - rho_target

    try:
        h_sol = brentq(f, h_min, h_max)
    except ValueError:
        # If solution not found in range, return None
        h_sol = None
    return h_sol


def atmospheric_properties(altitude_m):
    """
    Altitude-only hybrid atmosphere:
    - US Standard Atmosphere below 79 km
    - Exponential thermosphere above 79 km
    """

    gamma = 1.4
    R = 287.05  # J/(kg·K)

    # ----------------------------
    # Lower atmosphere (USSA)
    # ----------------------------
    if altitude_m <= 79_000.0:
        rho = density(altitude_m)
        temp = temperature(altitude_m)
        sound = np.sqrt(gamma * R * temp)

        if not np.isfinite(rho) or rho < 0:
            rho = 0.0
        if not np.isfinite(temp) or temp <= 0:
            temp = 200.0

        return rho, temp, sound

    # ----------------------------
    # Upper atmosphere (Exponential)
    # ----------------------------
    h_transition = 79_000.0

    # Anchor slightly BELOW transition for numerical stability
    h_anchor = 77_000.0
    rho0 = density(h_anchor)
    T0 = temperature(h_anchor)

    # Effective scale height (realistic for 80–120 km)
    H = 20_000.0  # meters

    rho = rho0 * np.exp(-(altitude_m - h_anchor) / H)

    # Thermospheric temperature profile (used for density shaping only)
    T_inf = 900.0  # K
    temp = T_inf - (T_inf - T0) * np.exp(-(altitude_m - h_anchor) / 40_000.0)

    # Freeze speed of sound above transition
    sound = np.sqrt(gamma * R * T0)

    return rho, temp, sound




class Spacecraft:
    def __init__(self, cl, cd, A, m, max_qc=None, nose_radius=3):
        self.cl = cl
        self.cd = cd
        self.Area = A
        self.mass = m
        self.banking_angle = 0
        self.nose_R = nose_radius
        self.max_qc = max_qc



    def initial_conditions(
        self,
        altitude,
        longitude,
        latitude,
        radial_velocity,
        tangential_velocity,
        inclination,
        planet_radius=6_371_000.0
    ):
        """
        altitude             : m above planet radius
        longitude, latitude  : radians
        inclination           : radians (0 = east, pi/2 = north)
        radial_velocity       : m/s (positive outward)
        tangential_velocity   : m/s (local horizontal magnitude)
        planet_radius         : m
        """

        # Store scalars
        self.altitude = altitude
        self.longitude = longitude
        self.latitude = latitude

        lat = latitude
        lon = longitude
        i = inclination

        # --- Local unit vectors (ECEF) ---

        # Radial (up)
        e_r = np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat)
        ])

        # East
        e_east = np.array([
            -np.sin(lon),
             np.cos(lon),
             0.0
        ])

        # North
        e_north = np.array([
            -np.sin(lat) * np.cos(lon),
            -np.sin(lat) * np.sin(lon),
             np.cos(lat)
        ])

        # Tangential direction in local horizontal plane
        e_t = np.cos(i) * e_east + np.sin(i) * e_north

        # --- Cartesian position ---
        r = planet_radius + altitude
        self.position_vector = r * e_r

        # --- Cartesian velocity ---
        self.cart_velocity_vector = (
            radial_velocity * e_r +
            tangential_velocity * e_t
        )
        self.cart_velocity_vector_og = self.v_inertial_toSOG(self.cart_velocity_vector)

    def keplerian_initial_conditions(
            self,
            apogee,
            perigee,
            altitude,
            inclination,
            arg_perigee,
            raan,
            true_anomaly_sign=1,
            planet_radius=6_371_000.0,
            mu=3.986004418e14
    ):
        """
        apogee, perigee : m above surface
        altitude        : current altitude above surface (m)
        inclination     : rad
        arg_perigee     : rad
        raan            : rad (longitude of ascending node)
        true_anomaly_sign : +1 ascending, -1 descending
        planet_radius   : m
        mu              : m^3/s^2
        """

        # --- Convert to orbital radii ---
        ra = planet_radius + apogee
        rp = planet_radius + perigee
        r = planet_radius + altitude

        # --- Semi-major axis and eccentricity ---
        a = 0.5 * (ra + rp)
        e = (ra - rp) / (ra + rp)

        # --- True anomaly from conic equation ---
        cos_nu = (a * (1 - e ** 2) / r - 1) / e
        cos_nu = np.clip(cos_nu, -1.0, 1.0)
        nu = true_anomaly_sign * np.arccos(cos_nu)

        # --- Specific angular momentum ---
        h = np.sqrt(mu * a * (1 - e ** 2))

        # --- Position & velocity in perifocal frame ---
        r_pf = np.array([
            r * np.cos(nu),
            r * np.sin(nu),
            0.0
        ])

        v_pf = np.array([
            -mu / h * np.sin(nu),
            mu / h * (e + np.cos(nu)),
            0.0
        ])

        # --- Rotation matrices ---
        def R3(theta):
            return np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

        def R1(theta):
            return np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])

        # --- Perifocal → ECI transformation ---
        Q = R3(raan) @ R1(inclination) @ R3(arg_perigee)

        self.position_vector = Q @ r_pf
        self.cart_velocity_vector = Q @ v_pf
        self.cart_velocity_vector_og = self.v_inertial_toSOG(self.cart_velocity_vector)

        # Store orbital elements
        self.apogee = apogee
        self.perigee = perigee
        self.altitude = altitude
        self.inclination = inclination
        self.arg_perigee = arg_perigee
        self.raan = raan
        self.true_anomaly = nu

    def v_inertial_toSOG(self, v_in, r_vec=None):
        """
        Convert inertial velocity to speed over ground (SOG).

        Parameters
        ----------
        v_in : ndarray
            3D velocity in ECI (m/s)
        r_vec : ndarray
            3D position in ECI (m)

        Returns
        -------
        v_sog : float
            Speed over ground (m/s)
        """

        if r_vec is None:
            r_vec = self.position_vector

        # Earth's rotation rate (rad/s)
        omega = 2 * np.pi / 86164.0  # sidereal day

        # Earth rotation vector
        k_hat = np.array([0.0, 0.0, 1.0])

        # Velocity of ground due to rotation
        v_earth = omega * np.cross(k_hat, r_vec)

        # Relative velocity over ground
        v_sog_vec = v_in - v_earth

        return v_sog_vec

    def aero_accelerations(self, r_vec, v_vec, planet_radius=6_371_000.0, mu=3.986004418e14):
        """
        Returns the Cartesian acceleration vectors (ECI) due to:
            - Gravity
            - Drag
            - Lift

        Returns:
            a_g : gravity acceleration vector (m/s^2)
            a_d : drag acceleration vector (m/s^2)
            a_l : lift acceleration vector (m/s^2)
        """

        # --- Position & velocity ---
        # r_vec = self.position_vector
        # v_vec = self.cart_velocity_vector

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        v_hat = v_vec / v

        # --- Altitude above surface ---
        self.altitude = r - planet_radius

        # --- Atmospheric density from COESA-76 ---
        rho, temp, sound = atmospheric_properties(self.altitude)

        self.mach = v/sound

        # --- Gravity acceleration vector ---
        a_g = -mu / r ** 3 * r_vec

        # --- Drag acceleration vector ---

        self.dynamic_pressure = 0.5 * rho * v ** 2


        if np.isnan(self.dynamic_pressure):
            F_d=0.0
        else:
            F_d = self.dynamic_pressure * self.cd * self.Area
        a_d = -F_d / self.mass * v_hat

        # --- Lift acceleration vector ---
        # Local vertical
        r_hat = r_vec / r

        # Orbit normal
        h_hat = np.cross(r_hat, v_hat)
        h_norm = np.linalg.norm(h_hat)
        if h_norm < 1e-8:
            h_hat = np.array([0.0, 0.0, 1.0])  # avoid zero vector
        else:
            h_hat /= h_norm

        # Nominal lift direction (perpendicular to velocity)
        lift_hat_0 = np.cross(v_hat, h_hat)
        lift_hat_0 /= np.linalg.norm(lift_hat_0)

        # Rotate lift vector by banking angle about velocity
        phi = self.banking_angle
        lift_hat = np.cos(phi) * lift_hat_0 + np.sin(phi) * h_hat

        # Lift acceleration vector

        if np.isnan(self.dynamic_pressure):
            F_l=0.0
        else:
            F_l = self.dynamic_pressure * self.cl * self.Area
        a_l = F_l / self.mass * lift_hat

        self.a_g = a_g
        self.a_d = a_d
        self.a_l = a_l

        self.a = a_g + a_d + a_l

        # Stagnation heat flux (Sutton Grave's equation) https://tfaws.nasa.gov/TFAWS12/Proceedings/Aerothermodynamics%20Course.pdf

        k = 1.7415e-4  # (Earth)
        # k = 1.9027e-4  # (Mars)

        self.qs = k * np.sqrt(rho/self.nose_R) * v ** 3

        if np.isnan(self.qs):
            self.qs=0


        return self.a

    def Euler_Rich_step(self, dt=1):
        an = self.aero_accelerations(self.position_vector, self.cart_velocity_vector_og)
        vn = self.cart_velocity_vector
        yn = self.position_vector

        v_mid = vn + 0.5*dt*an
        y_mid = yn + 0.5*dt*vn


        a_mid = self.aero_accelerations(y_mid, self.v_inertial_toSOG(v_mid))

        v_next = vn + dt*a_mid
        y_next = yn + dt*v_mid

        # --- Local vertical unit vector (north-up) ---
        e_r = y_next / np.linalg.norm(y_next)  # points away from Earth's center

        # --- Descent rate in m/s ---
        self.descend_rate = -np.dot(v_next, e_r)  # positive when descending

        self.cart_velocity_vector = v_next
        self.position_vector = y_next
        self.cart_velocity_vector_og = self.v_inertial_toSOG(self.cart_velocity_vector)


    def banking_angle_dr_P_controller(self, targetDR, kP=1.5):
        self.targetDR = targetDR
        error =  self.descend_rate - targetDR
        a_req = kP * error
        altitude = self.altitude
        rho, temp, sound = atmospheric_properties(altitude)

        if np.isnan(rho):
            rho=0

        v = np.linalg.norm(self.cart_velocity_vector_og)
        available_acc = 0.5 * rho * v ** 2 * self.cl * self.Area / self.mass


        if a_req > available_acc:
            self.banking_angle = 0
        elif -a_req > available_acc:
            self.banking_angle = np.pi
        else:
            self.banking_angle = np.arccos(a_req/available_acc)

    def banking_angle_h_P_controller_smart(self, kP_DR=1.5, kP_h = 0.02):

        # Firstly I'll compute the 0 DR required density
        v = np.linalg.norm(self.cart_velocity_vector_og)
        rho_req = self.mass*np.linalg.norm(self.a_g)/(0.5* v**2 * self.cl * self.Area)

        # Then I find at what altitude do I get that density
        self.target_altitude = altitude_for_density(rho_req)



        altitude_error = self.altitude - self.target_altitude
        DR = kP_h * altitude_error

        if self.target_altitude is None:
            if rho_req>1.22:
                DR=0
                self.targetDR = DR
                self.banking_angle = 0
            if rho_req<0.000001:
                DR=100
                self.banking_angle_dr_P_controller(DR, kP=kP_DR)
        else:
            self.banking_angle_dr_P_controller(DR, kP=kP_DR)






    def run_reentry(self, gif=True, controller=None, plot=True, dt=1.0, planet_radius=6_371_000.0, mu=3.986004418e14, gif_name="reentry.gif"):
        """
        Simulates reentry until altitude < 1 km.
        Records:
            - Altitude
            - Speed
            - Mach
            - Banking angle
            - g-force
            - Descent rate
            - Position
        Generates:
            - Time-series plots (altitude, speed, Mach, banking, g-force, descent rate vs time)
            - Speed and Mach vs altitude
            - 3D trajectory plot
            - Animated GIF of graphs & trajectory
        """
        initial_altitude = self.altitude
        # --- Initialization ---
        t = 0.0
        times = []
        altitudes = []
        speeds = []
        machs = []
        bank_angles = []
        g_forces = []
        descent_rates = []
        positions = []
        target_altitudes = []
        targetDRs = []
        dynamic_pressures = []
        heat_fluxes = []
        heat_loads = []

        heat_load = 0.0  # J/m^2 (integral of heat flux)

        max_steps = 50000  # safeguard
        step = 0

        g0 = 9.80665  # m/s^2
        prev_altitude = initial_altitude

        while True:
            r_vec = self.position_vector
            v_vec = self.cart_velocity_vector
            v_vec_og = self.cart_velocity_vector_og
            altitude = np.linalg.norm(r_vec) - planet_radius

            # Stop condition
            if altitude < 1000.0 or step >= max_steps:
                break

            # Take a step
            self.Euler_Rich_step(dt)

            # --- Thermal & aero quantities ---
            q_dyn = self.dynamic_pressure  # Pa
            q_dot = self.qs  # W/m^2 (stagnation heat flux)
            if q_dot > 0.0:
                heat_load += q_dot * dt  # J/m^2 (time integral)

            dynamic_pressures.append(q_dyn)
            heat_fluxes.append(q_dot)
            heat_loads.append(heat_load)

            if controller=="PDR":
                if altitude < 50000.0:
                    descend_rate = 0
                else:
                    descend_rate = 5

                self.banking_angle_dr_P_controller(descend_rate)
            elif controller=="PH":
                if self.altitude < 7000.0:
                    self.banking_angle = 0
                else:
                    self.banking_angle_h_P_controller_smart()


            # Compute descent rate
            descent_rate = self.descend_rate

            # Record state
            times.append(t)
            altitudes.append(altitude)
            speed = np.linalg.norm(v_vec_og)
            speeds.append(speed)
            machs.append(self.mach)
            bank_angles.append(np.rad2deg(self.banking_angle))  # degrees
            g = np.linalg.norm(self.a_l + self.a_d) / g0
            g_forces.append(g)
            descent_rates.append(descent_rate)
            positions.append(r_vec.copy())
            # Record target values if available
            if hasattr(self, 'target_altitude') and self.target_altitude is not None:
                target_altitudes.append(self.target_altitude)
            else:
                target_altitudes.append(np.nan)

            if hasattr(self, 'targetDR') and self.targetDR is not None:
                targetDRs.append(self.targetDR)
            else:
                targetDRs.append(np.nan)

            # Update time
            t += dt
            step += 1

            percent_down = 100 * (initial_altitude - altitude) / initial_altitude
            print(
                f"\rTime: {t:.1f}s | Alt: {altitude / 1000:.2f} km | Mach: {self.mach:.2f} | Bank: {np.rad2deg(self.banking_angle):.1f}° | g: {g:.2f}g | Descent: {descent_rate:.1f} m/s | {percent_down:.2f}% down",
                end='', flush=True)

        # Convert to arrays
        times = np.array(times)
        altitudes = np.array(altitudes)
        speeds = np.array(speeds)
        machs = np.array(machs)
        bank_angles = np.array(bank_angles)
        g_forces = np.array(g_forces)
        descent_rates = np.array(descent_rates)
        positions = np.array(positions)  # shape: (N, 3)
        target_altitudes = np.array(target_altitudes)
        targetDRs = np.array(targetDRs)
        dynamic_pressures = np.array(dynamic_pressures)
        heat_fluxes = np.array(heat_fluxes)
        heat_loads = np.array(heat_loads)

        if plot:
            # Create figure with gridspec for 2D + larger ground track
            fig = plt.figure(figsize=(20, 28))
            # Last row larger, top rows smaller to center the ground track vertically
            gs = gridspec.GridSpec(6, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1, 3])

            # 2D subplots
            ax_alt = fig.add_subplot(gs[0, 0])
            ax_speed = fig.add_subplot(gs[0, 1])
            ax_mach = fig.add_subplot(gs[1, 0])
            ax_speed_alt = fig.add_subplot(gs[1, 1])
            ax_mach_alt = fig.add_subplot(gs[2, 0])
            ax_bank = fig.add_subplot(gs[2, 1])
            ax_g = fig.add_subplot(gs[3, 0])
            ax_descent = fig.add_subplot(gs[3, 1])
            ax_qdyn = fig.add_subplot(gs[4, 0])
            ax_heat = fig.add_subplot(gs[4, 1])
            ax_heat_load = ax_heat.twinx()

            # Last row: ground track spanning full width
            ax_gt = fig.add_subplot(gs[5, :], projection=ccrs.PlateCarree())

            # --- Altitude vs Time ---
            ax_alt.plot(times, altitudes, 'r', label="Altitude")
            if np.any(~np.isnan(target_altitudes)):
                ax_alt.plot(times, target_altitudes, 'r--', label="Target Altitude")
            ax_alt.set_xlabel("Time (s)")
            ax_alt.set_ylabel("Altitude (m)")
            ax_alt.set_title("Altitude vs Time")
            ax_alt.legend()

            # --- Speed vs Time ---
            ax_speed.plot(times, speeds, 'b')
            ax_speed.set_xlabel("Time (s)")
            ax_speed.set_ylabel("Speed (m/s)")
            ax_speed.set_title("Speed vs Time")

            # --- Mach vs Time ---
            ax_mach.plot(times, machs, 'g')
            ax_mach.set_xlabel("Time (s)")
            ax_mach.set_ylabel("Mach")
            ax_mach.set_title("Mach vs Time")

            # --- Speed vs Altitude ---
            ax_speed_alt.plot(altitudes, speeds, 'b')
            ax_speed_alt.set_xlabel("Altitude (m)")
            ax_speed_alt.set_ylabel("Speed (m/s)")
            ax_speed_alt.set_title("Speed vs Altitude")

            # --- Mach vs Altitude ---
            ax_mach_alt.plot(altitudes, machs, 'g')
            ax_mach_alt.set_xlabel("Altitude (m)")
            ax_mach_alt.set_ylabel("Mach")
            ax_mach_alt.set_title("Mach vs Altitude")

            # --- Banking vs Time ---
            ax_bank.plot(times, bank_angles, 'm')
            ax_bank.set_xlabel("Time (s)")
            ax_bank.set_ylabel("Bank (deg)")
            ax_bank.set_title("Banking Angle vs Time")

            # --- g-force vs Time ---
            ax_g.plot(times, g_forces, 'c')
            ax_g.set_xlabel("Time (s)")
            ax_g.set_ylabel("g")
            ax_g.set_title("g-force vs Time")

            # --- Descent Rate vs Time ---
            ax_descent.plot(times, descent_rates, 'k', label="Descent Rate")
            if np.any(~np.isnan(targetDRs)):
                ax_descent.plot(times, targetDRs, 'k--', label="Target DR")
            ax_descent.set_xlabel("Time (s)")
            ax_descent.set_ylabel("Descent rate (m/s)")
            ax_descent.set_title("Descent Rate vs Time")
            ax_descent.legend()

            ax_qdyn.plot(times, dynamic_pressures / 1e3, color="orange")
            ax_qdyn.set_xlabel("Time (s)")
            ax_qdyn.set_ylabel("Dynamic Pressure (kPa)")
            ax_qdyn.set_title("Dynamic Pressure vs Time")

            # Heat flux (left axis)
            ax_heat.plot(
                times,
                heat_fluxes / 1e4,
                color="red",
                label="Heat Flux"
            )

            # Heat load (right axis)
            ax_heat_load.plot(
                times,
                heat_loads / 1e7,
                color="black",
                label="Heat Load"
            )

            ax_heat.fill_between(
                times,
                0,
                heat_fluxes / 1e4,
                color="red",
                alpha=0.25
            )

            ax_heat.set_xlabel("Time (s)")
            ax_heat.set_ylabel("Heat Flux (×1e4 W/m²)", color="red")
            ax_heat_load.set_ylabel("Heat Load (×1e7 J/m²)", color="black")

            ax_heat.set_title("Heat Flux & Integrated Heat Load vs Time")

            # --- Max heat flux ---
            idx_qdot = np.argmax(heat_fluxes)
            ax_heat.annotate(
                f"Max Heat Flux\n{heat_fluxes[idx_qdot]:.2e} W/m²",
                xy=(times[idx_qdot], heat_fluxes[idx_qdot] / 1e4),
                xytext=(10, 20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red"
            )

            # --- Horizontal line at max heat flux ---
            ax_heat.axhline(
                self.max_qc / 1e4,
                color="red",
                linestyle=":",
                linewidth=2,
                alpha=0.8
            )

            # --- Max heat load ---
            idx_qload = np.argmax(heat_loads)
            ax_heat_load.annotate(
                f"Max Heat Load\n{heat_loads[idx_qload]:.2e} J/m²",
                xy=(times[idx_qload], heat_loads[idx_qload] / 1e7),
                xytext=(-120, -30),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black"),
                color="black"
            )

            # --- Ground Track with altitude color ---
            lon, lat = eci_to_lonlat(positions, times)
            sc = ax_gt.scatter(
                np.rad2deg(lon), np.rad2deg(lat),
                c=altitudes, cmap='plasma', s=30,
                transform=ccrs.Geodetic()
            )
            ax_gt.stock_img()
            ax_gt.add_feature(cfeature.LAND, facecolor='lightgray')
            ax_gt.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax_gt.add_feature(cfeature.COASTLINE)
            ax_gt.set_title("Ground Track with Altitude")
            ax_gt.set_xlabel("Longitude (deg)")
            ax_gt.set_ylabel("Latitude (deg)")

            # Colorbar for altitude
            cbar = plt.colorbar(sc, ax=ax_gt, orientation='vertical', fraction=0.03, pad=0.02)
            cbar.set_label("Altitude (m)")

            plt.tight_layout()
            plt.show()

        # ----------------------------
        # GIF Animation
        # ----------------------------
        if gif:
            max_frames = 50
            fps = 20
            interval = 50

            total_steps = len(times)
            if total_steps <= max_frames:
                frame_indices = np.arange(total_steps)
            else:
                frame_indices = np.linspace(0, total_steps - 1, max_frames, dtype=int)

            # --- Figure and GridSpec like the static plot ---
            # Create figure with gridspec for 2D + larger ground track
            fig = plt.figure(figsize=(20, 28))
            # Last row larger, top rows smaller to center the ground track vertically
            gs = gridspec.GridSpec(6, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1, 3])

            ax_alt = fig.add_subplot(gs[0, 0])
            ax_speed = fig.add_subplot(gs[0, 1])
            ax_mach = fig.add_subplot(gs[1, 0])
            ax_speed_alt = fig.add_subplot(gs[1, 1])
            ax_mach_alt = fig.add_subplot(gs[2, 0])
            ax_bank = fig.add_subplot(gs[2, 1])
            ax_g = fig.add_subplot(gs[3, 0])
            ax_descent = fig.add_subplot(gs[3, 1])
            ax_qdyn = fig.add_subplot(gs[4, 0])
            ax_heat = fig.add_subplot(gs[4, 1])
            ax_heat_load = ax_heat.twinx()

            # --- Max heat flux reference line ---
            qdot_max = self.max_qc / 1e4
            ax_heat.axhline(
                qdot_max,
                color="red",
                linestyle=":",
                linewidth=2,
                alpha=0.8
            )

            ax_gt = fig.add_subplot(gs[5, :], projection=ccrs.PlateCarree())

            # --- Pre-setup ground track ---
            lon, lat = eci_to_lonlat(positions, times)
            sc = ax_gt.scatter([], [], c=[], cmap='plasma', s=30, transform=ccrs.Geodetic())
            ax_gt.stock_img()
            ax_gt.add_feature(cfeature.LAND, facecolor='lightgray')
            ax_gt.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax_gt.add_feature(cfeature.COASTLINE)
            ax_gt.set_title("Ground Track with Altitude")
            ax_gt.set_xlabel("Longitude (deg)")
            ax_gt.set_ylabel("Latitude (deg)")
            cbar = plt.colorbar(sc, ax=ax_gt, orientation='vertical', fraction=0.03, pad=0.02)
            cbar.set_label("Altitude (m)")

            # --- Empty lines for animation ---
            line_alt, = ax_alt.plot([], [], 'r')
            line_speed, = ax_speed.plot([], [], 'b')
            line_mach, = ax_mach.plot([], [], 'g')
            line_speed_alt, = ax_speed_alt.plot([], [], 'b')
            line_mach_alt, = ax_mach_alt.plot([], [], 'g')
            line_bank, = ax_bank.plot([], [], 'm')
            line_g, = ax_g.plot([], [], 'c')
            line_descent, = ax_descent.plot([], [], 'k')
            line_qdyn, = ax_qdyn.plot([], [], color="orange")
            line_qdot, = ax_heat.plot([], [], color="red")
            line_qload, = ax_heat_load.plot([], [], color="black")

            def update(frame_idx):
                frame = frame_indices[frame_idx]
                print(f"\rRendering GIF frame {frame_idx + 1}/{len(frame_indices)}...", end='', flush=True)

                # 2D plots
                line_alt.set_data(times[:frame], altitudes[:frame])
                ax_alt.relim();
                ax_alt.autoscale_view()
                line_speed.set_data(times[:frame], speeds[:frame])
                ax_speed.relim();
                ax_speed.autoscale_view()
                line_mach.set_data(times[:frame], machs[:frame])
                ax_mach.relim();
                ax_mach.autoscale_view()
                line_speed_alt.set_data(altitudes[:frame], speeds[:frame])
                ax_speed_alt.relim();
                ax_speed_alt.autoscale_view()
                line_mach_alt.set_data(altitudes[:frame], machs[:frame])
                ax_mach_alt.relim();
                ax_mach_alt.autoscale_view()
                line_bank.set_data(times[:frame], bank_angles[:frame])
                ax_bank.relim();
                ax_bank.autoscale_view()
                line_g.set_data(times[:frame], g_forces[:frame])
                ax_g.relim();
                ax_g.autoscale_view()
                line_descent.set_data(times[:frame], descent_rates[:frame])
                ax_descent.relim();
                ax_descent.autoscale_view()
                line_qdyn.set_data(times[:frame], dynamic_pressures[:frame] / 1e3)
                ax_qdyn.relim()
                ax_qdyn.autoscale_view()

                line_qdot.set_data(
                    times[:frame],
                    heat_fluxes[:frame] / 1e4
                )

                line_qload.set_data(
                    times[:frame],
                    heat_loads[:frame] / 1e7
                )


                ax_heat.relim()
                ax_heat.autoscale_view()

                ax_heat_load.relim()
                ax_heat_load.autoscale_view()

                idx_qdot = np.argmax(heat_fluxes)
                idx_qload = np.argmax(heat_loads)

                ax_heat.annotate(
                    "Max Heat Flux",
                    xy=(times[idx_qdot], heat_fluxes[idx_qdot] / 1e4),
                    xytext=(10, 20),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="red"),
                    color="red"
                )

                ax_heat_load.annotate(
                    "Max Heat Load",
                    xy=(times[idx_qload], heat_loads[idx_qload] / 1e7),
                    xytext=(-120, -30),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black"),
                    color="black"
                )

                # Ground track
                sc.set_offsets(np.column_stack((np.rad2deg(lon[:frame]), np.rad2deg(lat[:frame]))))
                sc.set_array(altitudes[:frame])

                return (line_alt, line_speed, line_mach,
                        line_speed_alt, line_mach_alt, line_bank,
                        line_g, line_descent, sc, line_qdyn, line_qdot, line_qload)

            anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval, blit=False)
            # writer = PillowWriter(fps=fps) # Pillow
            writer = FFMpegWriter(fps=20) # FFMpeg

            anim.save("reentry.mp4", writer=writer) # FFMpeg
            # anim.save(gif_name, writer=writer) # Pillow

            print("\r" + " " * 120 + "\r", end='')  # clear loading line
            print(f"Reentry animation saved as {gif_name}")

        return times, altitudes, speeds, machs, bank_angles, g_forces, descent_rates, positions

    def plot_orbit_3d_init(
            self,
            apogee,
            perigee,
            inclination,
            arg_perigee,
            raan,
            planet_radius=6_371_000.0,
            mu=3.986004418e14,
            num_points=500,
            velocity_scale=0.1
    ):
        """
        Plots the full orbit, spacecraft position, and velocity vector in 3D (ECI frame)

        velocity_scale : fraction of planet radius used for arrow length
        """

        # --- Orbital parameters ---
        ra = planet_radius + apogee
        rp = planet_radius + perigee
        a = 0.5 * (ra + rp)
        e = (ra - rp) / (ra + rp)

        # True anomaly range
        nu = np.linspace(0, 2 * np.pi, num_points)

        # Radius as function of true anomaly
        r = a * (1 - e ** 2) / (1 + e * np.cos(nu))

        # Perifocal coordinates
        r_pf = np.vstack((
            r * np.cos(nu),
            r * np.sin(nu),
            np.zeros_like(nu)
        ))

        # Rotation matrices
        def R3(theta):
            return np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

        def R1(theta):
            return np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])

        # Perifocal → ECI transformation
        Q = R3(raan) @ R1(inclination) @ R3(arg_perigee)
        r_eci = Q @ r_pf

        # --- Plot ---
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")

        # Planet
        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        x = planet_radius * np.cos(u) * np.sin(v)
        y = planet_radius * np.sin(u) * np.sin(v)
        z = planet_radius * np.cos(v)
        ax.plot_surface(x, y, z, alpha=0.3)

        # Orbit
        ax.plot(
            r_eci[0], r_eci[1], r_eci[2],
            label="Orbit"
        )

        # Spacecraft position
        r_sc = self.position_vector
        v_sc = self.cart_velocity_vector

        ax.scatter(
            r_sc[0], r_sc[1], r_sc[2],
            s=60,
            label="Spacecraft"
        )

        # --- Velocity vector ---
        v_hat = v_sc / np.linalg.norm(v_sc)
        arrow_length = velocity_scale * planet_radius

        ax.quiver(
            r_sc[0], r_sc[1], r_sc[2],
            v_hat[0], v_hat[1], v_hat[2],
            length=arrow_length,
            normalize=True,
            linewidth=2,
            label="Velocity"
        )

        # Formatting
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Orbit, Spacecraft Position, and Velocity Vector")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])

        plt.show()

    def plot_orbit_3d(
            self,
            planet_radius=6_371_000.0,
            mu=3.986004418e14,
            num_points=500,
            velocity_scale=0.1
    ):
        """
        Plots the orbit using ONLY the current Cartesian state:
        - self.position_vector (ECI)
        - self.cart_velocity_vector (ECI)
        """

        import numpy as np
        import matplotlib.pyplot as plt

        r_vec = self.position_vector
        v_vec = self.cart_velocity_vector

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        # ----------------------------
        # Orbital elements from state
        # ----------------------------
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)

        e_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec / r)
        e = np.linalg.norm(e_vec)

        a = 1 / (2 / r - v ** 2 / mu)

        i = np.arccos(h_vec[2] / h)

        K = np.array([0.0, 0.0, 1.0])
        n_vec = np.cross(K, h_vec)
        n = np.linalg.norm(n_vec)

        raan = np.arctan2(n_vec[1], n_vec[0])

        arg_perigee = np.arccos(
            np.dot(n_vec, e_vec) / (n * e)
        ) if e > 1e-8 else 0.0

        if e_vec[2] < 0:
            arg_perigee = 2 * np.pi - arg_perigee

        # ----------------------------
        # Reconstruct orbit
        # ----------------------------
        nu = np.linspace(0, 2 * np.pi, num_points)
        p = a * (1 - e ** 2)
        r_orbit = p / (1 + e * np.cos(nu))

        r_pf = np.vstack((
            r_orbit * np.cos(nu),
            r_orbit * np.sin(nu),
            np.zeros_like(nu)
        ))

        # Rotation matrices
        def R3(theta):
            return np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

        def R1(theta):
            return np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])

        Q = R3(raan) @ R1(i) @ R3(arg_perigee)
        r_eci = Q @ r_pf

        # ----------------------------
        # Plot
        # ----------------------------
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")

        # Planet
        u_sph, v_sph = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        x = planet_radius * np.cos(u_sph) * np.sin(v_sph)
        y = planet_radius * np.sin(u_sph) * np.sin(v_sph)
        z = planet_radius * np.cos(v_sph)
        ax.plot_surface(x, y, z, alpha=0.3)

        # Orbit
        ax.plot(r_eci[0], r_eci[1], r_eci[2], label="Orbit")

        # Spacecraft position
        ax.scatter(
            r_vec[0], r_vec[1], r_vec[2],
            s=60, label="Spacecraft"
        )

        # Velocity vector
        v_hat = v_vec / v
        arrow_length = velocity_scale * planet_radius

        ax.quiver(
            r_vec[0], r_vec[1], r_vec[2],
            v_hat[0], v_hat[1], v_hat[2],
            length=arrow_length,
            normalize=True,
            linewidth=2,
            label="Velocity"
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Orbit from Cartesian State (ECI)")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])

        # ----------------------------
        # Banking (lift) vector
        # ----------------------------
        r_hat = r_vec / np.linalg.norm(r_vec)
        v_hat = v_vec / np.linalg.norm(v_vec)

        h_hat = np.cross(r_hat, v_hat)
        h_hat /= np.linalg.norm(h_hat)

        lift_hat_0 = np.cross(v_hat, h_hat)
        lift_hat_0 /= np.linalg.norm(lift_hat_0)

        phi = self.banking_angle  # radians

        lift_hat = (
                np.cos(phi) * lift_hat_0 +
                np.sin(phi) * h_hat
        )

        lift_length = 0.08 * planet_radius

        ax.quiver(
            r_vec[0], r_vec[1], r_vec[2],
            lift_hat[0], lift_hat[1], lift_hat[2],
            length=lift_length,
            normalize=True,
            linewidth=2,
            linestyle="dashed",
            label="Bank / Lift",
            color="green"
        )

        plt.show()

