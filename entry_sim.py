
import numpy as np
import matplotlib.pyplot as plt
from pystdatm import density, temperature, speed_of_sound
from matplotlib.animation import FuncAnimation, PillowWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def ecef_to_lonlat(r_vec, planet_radius=6_371_000.0):
    """
    Converts an ECEF Cartesian position vector to geodetic latitude and longitude (radians)
    Assumes spherical Earth.
    """
    x, y, z = r_vec.T  # shape: (N,3)
    lon = np.arctan2(y, x)
    lat = np.arcsin(z / np.linalg.norm(r_vec, axis=1))
    return lon, lat


def atmospheric_properties(altitude_m):
    rho = density(altitude_m)
    temp = temperature(altitude_m)
    sound = speed_of_sound(temp)
    return rho, temp, sound

class Spacecraft:
    def __init__(self, cl, cd, A, m):
        self.cl = cl
        self.cd = cd
        self.Area = A
        self.mass = m
        self.banking_angle = 0



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

        # Store orbital elements
        self.apogee = apogee
        self.perigee = perigee
        self.altitude = altitude
        self.inclination = inclination
        self.arg_perigee = arg_perigee
        self.raan = raan
        self.true_anomaly = nu

    def aero_accelerations(self, r_vec, v_vec, planet_radius=6_371_000.0, mu=3.986004418e14):
        """
        Returns the Cartesian acceleration vectors (ECI) due to:
            - Gravity
            - Drag
            - Lift

        Uses COESA-76 standard atmosphere (via pyatmos) for density.

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
        F_d = 0.5 * rho * v ** 2 * self.cd * self.Area
        if np.isnan(F_d):
            F_d=0.0
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
        F_l = 0.5 * rho * v ** 2 * self.cl * self.Area
        if np.isnan(F_l):
            F_l=0.0
        a_l = F_l / self.mass * lift_hat

        self.a_g = a_g
        self.a_d = a_d
        self.a_l = a_l

        self.a = a_g + a_d + a_l


        return self.a

    def Euler_Rich_step(self, dt=1):
        an = self.aero_accelerations(self.position_vector, self.cart_velocity_vector)
        vn = self.cart_velocity_vector
        yn = self.position_vector

        v_mid = vn + 0.5*dt*an
        y_mid = yn + 0.5*dt*vn

        a_mid = self.aero_accelerations(y_mid, v_mid)

        v_next = vn + dt*a_mid
        y_next = yn + dt*v_mid

        # --- Local vertical unit vector (north-up) ---
        e_r = y_next / np.linalg.norm(y_next)  # points away from Earth's center

        # --- Descent rate in m/s ---
        self.descend_rate = -np.dot(v_next, e_r)  # positive when descending

        self.cart_velocity_vector = v_next
        self.position_vector = y_next

    def banking_angle_dr_P_controller(self, targetDR, kP=1.85):
        error =  self.descend_rate - targetDR
        a_req = kP * error
        altitude = self.altitude
        if altitude > 79000:
            altitude = 79000
        rho, temp, sound = atmospheric_properties(altitude)

        if np.isnan(rho):
            rho=0

        v = np.linalg.norm(self.cart_velocity_vector)
        available_acc = 0.5 * rho * v ** 2 * self.cl * self.Area / self.mass


        if a_req > available_acc:
            self.banking_angle = 0
        elif -a_req > available_acc:
            self.banking_angle = np.pi
        else:
            self.banking_angle = np.arccos(a_req/available_acc)



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

        max_steps = 50000  # safeguard
        step = 0

        g0 = 9.80665  # m/s^2
        prev_altitude = initial_altitude

        while True:
            r_vec = self.position_vector
            v_vec = self.cart_velocity_vector
            altitude = np.linalg.norm(r_vec) - planet_radius

            # Stop condition
            if altitude < 1000.0 or step >= max_steps:
                break

            # Take a step
            self.Euler_Rich_step(dt)
            if controller=="PDR":
                if altitude < 50000.0:
                    descend_rate = 0
                else:
                    descend_rate = 5

                self.banking_angle_dr_P_controller(descend_rate)


            # Compute descent rate
            descent_rate = self.descend_rate

            # Record state
            times.append(t)
            altitudes.append(altitude)
            speed = np.linalg.norm(v_vec)
            speeds.append(speed)
            machs.append(self.mach)
            bank_angles.append(np.rad2deg(self.banking_angle))  # degrees
            g = np.linalg.norm(self.a_l + self.a_d) / g0
            g_forces.append(g)
            descent_rates.append(descent_rate)
            positions.append(r_vec.copy())

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
        if plot:
            # ----------------------------
            # STATIC PLOTS
            # ----------------------------
            fig, axes = plt.subplots(5, 2, figsize=(14, 14))
            plt.tight_layout(pad=4.0)
            ax_alt = axes[0, 0]
            ax_speed = axes[0, 1]
            ax_mach = axes[1, 0]
            ax_speed_alt = axes[1, 1]
            ax_mach_alt = axes[2, 0]
            ax_bank = axes[2, 1]
            ax_g = axes[3, 0]
            ax_descent = axes[3, 1]
            ax_3d = fig.add_subplot(5, 2, 10, projection='3d')

            ax_alt.plot(times, altitudes, 'r')
            ax_alt.set_xlabel("Time (s)")
            ax_alt.set_ylabel("Altitude (m)")
            ax_alt.set_title("Altitude vs Time")

            ax_speed.plot(times, speeds, 'b')
            ax_speed.set_xlabel("Time (s)")
            ax_speed.set_ylabel("Speed (m/s)")
            ax_speed.set_title("Speed vs Time")

            ax_mach.plot(times, machs, 'g')
            ax_mach.set_xlabel("Time (s)")
            ax_mach.set_ylabel("Mach")
            ax_mach.set_title("Mach vs Time")

            ax_speed_alt.plot(altitudes, speeds, 'b')
            ax_speed_alt.set_xlabel("Altitude (m)")
            ax_speed_alt.set_ylabel("Speed (m/s)")
            ax_speed_alt.set_title("Speed vs Altitude")

            ax_mach_alt.plot(altitudes, machs, 'g')
            ax_mach_alt.set_xlabel("Altitude (m)")
            ax_mach_alt.set_ylabel("Mach")
            ax_mach_alt.set_title("Mach vs Altitude")

            ax_bank.plot(times, bank_angles, 'm')
            ax_bank.set_xlabel("Time (s)")
            ax_bank.set_ylabel("Bank (deg)")
            ax_bank.set_title("Banking Angle vs Time")

            ax_g.plot(times, g_forces, 'c')
            ax_g.set_xlabel("Time (s)")
            ax_g.set_ylabel("g")
            ax_g.set_title("g-force vs Time")

            ax_descent.plot(times, descent_rates, 'k')
            ax_descent.set_xlabel("Time (s)")
            ax_descent.set_ylabel("Descent rate (m/s)")
            ax_descent.set_title("Descent Rate vs Time")

            # 3D trajectory
            u, v_ = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
            x = planet_radius * np.cos(u) * np.sin(v_)
            y = planet_radius * np.sin(u) * np.sin(v_)
            z = planet_radius * np.cos(v_)
            ax_3d.plot_surface(x, y, z, alpha=0.3, color='b')
            ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r', lw=2)
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.set_title("3D Trajectory")
            ax_3d.set_box_aspect([1, 1, 1])

            plt.show()

            # ----------------------------
            # Ground track plot with world map
            # ----------------------------
            lon, lat = ecef_to_lonlat(positions)

            fig_gt = plt.figure(figsize=(12, 6))
            ax_gt = plt.axes(projection=ccrs.PlateCarree())
            ax_gt.stock_img()  # basic world map background
            ax_gt.add_feature(cfeature.LAND, facecolor='lightgray')
            ax_gt.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax_gt.add_feature(cfeature.COASTLINE)

            # Plot ground track
            ax_gt.plot(np.rad2deg(lon), np.rad2deg(lat), 'r', lw=2, transform=ccrs.Geodetic())
            ax_gt.set_title("Ground Track")
            ax_gt.set_xlabel("Longitude (deg)")
            ax_gt.set_ylabel("Latitude (deg)")
            plt.show()

        # ----------------------------
        # GIF Animation
        # ----------------------------
        if gif:
            from matplotlib.animation import FuncAnimation, PillowWriter
            max_frames = 50
            fps = 20
            interval = 50

            total_steps = len(times)
            if total_steps <= max_frames:
                frame_indices = np.arange(total_steps)
            else:
                frame_indices = np.linspace(0, total_steps - 1, max_frames, dtype=int)

            fig, axes = plt.subplots(5, 2, figsize=(14, 14))
            plt.tight_layout(pad=4.0)
            ax_alt = axes[0, 0]
            ax_speed = axes[0, 1]
            ax_mach = axes[1, 0]
            ax_speed_alt = axes[1, 1]
            ax_mach_alt = axes[2, 0]
            ax_bank = axes[2, 1]
            ax_g = axes[3, 0]
            ax_descent = axes[3, 1]
            ax_3d = fig.add_subplot(5, 2, 10, projection='3d')

            # Pre-draw planet
            ax_3d.plot_surface(x, y, z, alpha=0.3, color='b')
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.set_title("3D Trajectory")
            ax_3d.set_box_aspect([1, 1, 1])

            # Empty lines for animation
            line_alt, = ax_alt.plot([], [], 'r')
            line_speed, = ax_speed.plot([], [], 'b')
            line_mach, = ax_mach.plot([], [], 'g')
            line_speed_alt, = ax_speed_alt.plot([], [], 'b')
            line_mach_alt, = ax_mach_alt.plot([], [], 'g')
            line_bank, = ax_bank.plot([], [], 'm')
            line_g, = ax_g.plot([], [], 'c')
            line_descent, = ax_descent.plot([], [], 'k')
            traj_3d, = ax_3d.plot([], [], [], 'r', lw=2)

            def update(frame_idx):
                frame = frame_indices[frame_idx]
                print(f"\rRendering GIF frame {frame_idx + 1}/{len(frame_indices)}...", end='', flush=True)

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
                traj_3d.set_data(positions[:frame, 0], positions[:frame, 1])
                traj_3d.set_3d_properties(positions[:frame, 2])

                return (line_alt, line_speed, line_mach,
                        line_speed_alt, line_mach_alt, line_bank,
                        line_g, line_descent, traj_3d)

            anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval, blit=False)
            writer = PillowWriter(fps=fps)
            anim.save(gif_name, writer=writer)

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

# ------------------------------
# Starship parameters (empty)
# ------------------------------
cl = 1.2
cd = 1.3
area = 545.0       # m^2
mass = 120_000.0    # kg

# Create spacecraft
sc = Spacecraft(cl=cl, cd=cd, A=area, m=mass)

# ------------------------------
# Orbit definition
# ------------------------------
apogee = 213_000.0
perigee = -15_000.0
altitude = 100_000.0   # current altitude

inclination = np.deg2rad(26.8)     # Starship-like
arg_perigee = np.deg2rad(10.0)
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