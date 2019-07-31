from dolfin import *
from math import tanh
import time
import numpy as np
from mshr import *

class CH_InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def set_Cn(self, Cn):
        self.Cn = Cn

    def set_x_new_f(self, x_new, f):
        self.x_new = x_new
        self.f = f

    def eval(self, values, x):
        f = self.f
        if (f(x[1] + 8.5) * 0.98 > x[0]):
            values[0] = -1
        else:
            values[0] = 1

    def value_shape(self):
        return (2,)


class Problem:
    def __init__(self, Pe, Cn, Ca, lamb, theta1, theta2, dp):
        self.Pe = Pe
        self.Cn = Cn
        self.Ca = Ca
        self.lamb = lamb
        self.theta1 = theta1
        self.theta2 = theta2
        self.dp = dp

    def set_x_new_f(self, x_new, f):
        self.x_new = x_new
        self.f = f

    def load_mesh(self, mesh):
        self.nonrefinedmesh = mesh
        self.mesh = mesh

    def refine_mesh(self, nr_refinements, phi_boundary, first_case=False):
        mesh = self.nonrefinedmesh
        for i in range(nr_refinements):
            cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
            cell_markers.set_all(False)
            if not first_case:
                phi_value1, mu_value1 = self.CHsol.split()
                for c in cells(mesh):
                    coordinates = c.get_vertex_coordinates()
                    r_value = (coordinates[0])
                    z_value = (coordinates[1])
                    coord_move1 = self.Cn * 1
                    phi_value1_00 = abs(phi_value1(r_value, z_value))
                    try:
                        phi_value1_01p = abs(phi_value1(r_value, z_value + coord_move1))
                        phi_value1_01m = abs(phi_value1(r_value, z_value - coord_move1))
                    except:
                        phi_value1_01p = abs(phi_value1(r_value, z_value))
                        phi_value1_01m = abs(phi_value1(r_value, z_value))
                    if phi_value1_00 < phi_boundary:
                        cell_markers[c] = True
                    elif phi_value1_01p < phi_boundary:
                        cell_markers[c] = True
                    elif phi_value1_01m < phi_boundary:
                        cell_markers[c] = True
                    else:
                        cell_markers[c] = False
            # Refine mesh
            mesh = refine(mesh, cell_markers)
            print('refinement complete')
        self.mesh = mesh
        boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1,
                                      value=0)
        botwall_secondpart = CompiledSubDomain('x[0] > 0.3 && on_boundary')
        botwall_firstpart = CompiledSubDomain('x[0] > 0.3 && on_boundary')
        z_axis = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
        firstwall = CompiledSubDomain('near(x[1], -10) && on_boundary')
        secondwall = CompiledSubDomain('near(x[1], 7.5) && on_boundary')
        botwall_secondpart.mark(boundaries, 5)
        botwall_firstpart.mark(boundaries, 2)
        firstwall.mark(boundaries, 1)
        secondwall.mark(boundaries, 3)
        z_axis.mark(boundaries, 4)
        file = File(file_name + 'domain.pvd')
        file << boundaries
        self.ds = Measure("ds")(subdomain_data=boundaries)
        self.boundaries = boundaries

    def create_function_space_mesh(self):
        self.P1 = VectorElement('Lagrange', self.mesh.ufl_cell(), degree=2)
        self.P2 = FiniteElement('Lagrange', self.mesh.ufl_cell(), degree=1)
        self.P3 = FiniteElement('Lagrange', self.mesh.ufl_cell(), degree=1)
        self.P4 = FiniteElement('Lagrange', self.mesh.ufl_cell(), degree=1)
        self.SF = MixedElement([self.P1, self.P2])
        self.CHF = MixedElement([self.P3, self.P4])
        self.S = FunctionSpace(self.mesh, self.SF)
        self.CH = FunctionSpace(self.mesh, self.CHF)

        self.dSsol = TrialFunction(self.S)
        self.k, self.l = TestFunctions(self.S)

        self.dCHsol = TrialFunction(self.CH)
        self.q, self.v = TestFunctions(self.CH)

    def initialize_field(self):
        print('initializing')
        self.Ssol = Function(self.S)
        self.Ssol0 = Function(self.S)
        self.CHsol = Function(self.CH)
        self.CHsol0 = Function(self.CH)
        CHsol_init = CH_InitialConditions(degree=1)
        CHsol_init.set_Cn(self.Cn)
        CHsol_init.set_x_new_f(self.x_new, self.f)
        self.CHsol.interpolate(CHsol_init)
        self.CHsol0.interpolate(CHsol_init)
        print('initializing done')

    def interpolate_field(self):
        print('interpolating')
        self.Ssol0 = interpolate(self.Ssol0, self.S)
        self.CHsol0 = interpolate(self.CHsol0, self.CH)
        self.Ssol = interpolate(self.Ssol, self.S)
        self.CHsol = interpolate(self.CHsol, self.CH)
        print('interpolating done')

    def split_functions(self):
        self.u, self.p = split(self.Ssol)
        self.u0, self.p0 = split(self.Ssol0)
        self.phi, self.mu = split(self.CHsol)
        self.phi0, self.mu0 = split(self.CHsol0)

    def formulate_problem(self, Dt):

        bcu_z_axis = DirichletBC(self.S.sub(0).sub(0), Constant((0)),
                                 self.boundaries, 4)
        inflow_profile = ('0', '2 * (1 - x[0] * x[0])')

        # Define boundary conditions
        bcu_firstwall = DirichletBC(self.S.sub(0), Expression(inflow_profile, degree=2),
                                    self.boundaries, 1)
        bcu_botwall_firstpart = DirichletBC(self.S.sub(0), Constant((0, 0)), self.boundaries, 2)
        bcu_botwall_secondpart = DirichletBC(self.S.sub(0), Constant((0, 0)), self.boundaries, 5)
        bcu_secondwall = DirichletBC(self.S.sub(0), Expression(inflow_profile, degree=2),
                                    self.boundaries, 1)
        bcp_firstwall = DirichletBC(self.S.sub(1), Constant((self.dp)), self.boundaries, 1)
        bcp_botwall_firstpart = DirichletBC(self.S.sub(1), Constant((0)), self.boundaries, 2)
        bcp_secondwall = DirichletBC(self.S.sub(1), Constant((0)), self.boundaries, 3)

        bcu = [bcu_botwall_firstpart]
        bcp = [bcp_secondwall, bcp_firstwall]
        self.BCs = bcu + bcp
        self.phi = variable(self.phi)
        F = 0.25 * (self.phi**2 - 1)**2
        dFdphi = diff(F, self.phi)
        g = 0.5 + 0.75 * self.phi - 0.25 * self.phi**3
        dgdphi = diff(g, self.phi)
        eta = 0.5 * (1 + self.phi + self.lamb * (1 - self.phi))
        self.eta = eta
        x = SpatialCoordinate(self.mesh)
        r = x[0]
        self.r = r
        ##################
        # Stokes equation:
        # r - component
        L0 = -Dx(self.p, 0) * self.k[0] * r**2 * dx
        L0 -= eta * Dx(r * self.u[0], 0) * (2 * self.k[0] + r * Dx(self.k[0], 0)) * dx
        L0 -= eta * Dx(self.u[0], 1) * Dx(self.k[0] * r**2, 1) * dx
        # z - component
        L1 = -Dx(self.p, 1) * r * self.k[1] * dx
        L1 -= eta * r * Dx(self.u[1], 0) * Dx(self.k[1], 0) * dx
        L1 -= eta * Dx(self.u[1], 1) * Dx(self.k[1] * r, 1) * dx
        # continuity equation
        L2 = Dx(r * self.u[0], 0) * self.l * dx
        L2 += Dx(self.u[1], 1) * self.l * r * dx
        L = L0 + L1 + L2
        # Coupling terms:
        L += self.mu / (self.Cn * self.Ca) * Dx(self.phi, 0) * self.k[0] * r**2 * dx
        L += self.mu / (self.Cn * self.Ca) * Dx(self.phi, 1) * self.k[1] * r * dx
        self.SL = L
        # Cahn-Hillard:
        L = (self.phi - self.phi0) * self.q * r * dx
        L += Dt * self.u[0] * Dx(self.phi, 0) * self.q * r * dx
        L += Dt * self.u[1] * Dx(self.phi, 1) * self.q * r * dx
        L += Dt * 1 / self.Pe * r * Dx(self.mu, 0) * Dx(self.q, 0) * dx
        L += Dt * 1 / self.Pe * Dx(self.mu, 1) * r * Dx(self.q, 1) * dx
        L += Dt * self.mu * self.v * r * dx
        L -= Dt * dFdphi * self.v * r * dx
        L -= Dt * self.Cn**2 * r * Dx(self.phi, 0) * Dx(self.v, 0) * dx
        L -= Dt * self.Cn**2 * Dx(self.phi, 1) * r * Dx(self.v, 1) * dx
        # Contact angle boundary condition:
        L -= Dt * self.Cn * cos(self.theta1) * self.v * r * dgdphi * self.ds(2) # first part wetting condition
        L -= Dt * self.Cn * cos(self.theta2) * self.v * r * dgdphi * self.ds(5) # last part wetting condition
        self.CHL = L


def emulsion_shape_init():
    def make_fit(t, x, degree):
        # calculate polynomial
        z = np.polyfit(t, x, degree)
        f = np.poly1d(z)

        # calculate new x's and y's
        t_new = np.linspace(t[0], t[-1], 1000)
        x_new = f(t_new)
        return t_new, x_new, f

    boundary_file = droplet + '.txt'
    folder = '../'
    #p1 = 12
    p1 = 100
    if droplet == 'p1_100_p2_7_60fps_extracted_droplet_2':
        vert_off = 168
        hor_off = 210
        radius = 33
    elif droplet == 'p1_100_p2_7_60fps_extracted_droplet_3':
        vert_off = 168
        hor_off = 200
        radius = 33
    else:
        vert_off = 95
        hor_off = 250
        radius = 33


    x_file = []
    y_file = []
    with open(folder + boundary_file, 'r') as f:
        for line in f:
                x_file.append(float(line.split()[0]))
                y_file.append(float(line.split()[1]))

    x_file = np.asarray(x_file)
    y_file = vert_off - np.asarray(y_file)
    normalized_x = (x_file - hor_off) / radius
    normalized_y = y_file / radius

    x_new, y_new, f = make_fit(normalized_x[:], normalized_y[:], 10)
    return x_new, f


def f(z):
    if np.abs(z) > 2:
        func = 1
    else:
        func= -np.cos(z * np.pi / 2) * 0.1 - 0.1 + 1
    return func

def create_mesh_constrict(zmin, zmax, n, file_name):
    z = np.linspace(zmin, zmax, n)
    function = [f(z_) for z_ in z]
    bottom_wall = [Point(f(z_), z_) for z_ in z]

    top_wall = [Point(0, zmax), Point(0, 0), Point(0, zmin)]
    domain = Polygon(bottom_wall + top_wall)
    mesh = generate_mesh(domain, n)
    return mesh


def read_capillary_and_make_mesh(zmin=-10, zmax=20, n=200):
    boundary_file = 'constriction_extract_coords.txt'
    folder = '../'

    x_file = []
    y_file = []
    with open(folder + boundary_file, 'r') as f:
        for line in f:
                x_file.append(float(line.split()[0]))
                y_file.append(float(line.split()[1]))

    x_file = np.asarray(x_file)
    y_file = 56 - np.asarray(y_file)
    normalized_x = x_file / ((y_file[0] + y_file[-1]) / 2) - 14
    normalized_y = y_file / ((y_file[0] + y_file[-1]) / 2)

    def make_fit(t, x, degree):
        # calculate polynomial
        z = np.polyfit(t, x, degree)
        f = np.poly1d(z)

        # calculate new x's and y's
        t_new = np.linspace(t[0], t[-1], 100)
        x_new = f(t_new)
        return t_new, x_new, f

    bottom_wall = [Point(normalized_y[i], normalized_x[i]) for i in range(10)]
    normalized_x_fit, normalized_y_fit, f = make_fit(normalized_x[10:-14], normalized_y[10:-14], 12)
    bottom_wall += [Point(normalized_y_fit[i], normalized_x_fit[i]) for i in range(len(normalized_x_fit))]
    bottom_wall += [Point(normalized_y[i], normalized_x[i]) for i in range(len(normalized_x)-14, len(normalized_x))]
    top_wall = [Point(normalized_y[-1], zmax), Point(0, zmax),
                Point(0, normalized_x[-1]), Point(0, normalized_x[0]),
                Point(0, zmin), Point(normalized_y[0], zmin)]
    domain = Polygon(bottom_wall + top_wall)
    mesh = generate_mesh(domain, n)
    return mesh

def SCHsolve(nr=50, nz=100, Pe=3e-3, Cn=0.06, Ca=1e0, theta1=0, theta2=0, lamb=1.0,
             alpha=1.0, nr_steps=25, dp = 10, write_to_file=False, file_name='', Dt=0.1):
    start_time = time.time()
    print('Cn =', Cn, 'Pe =', Pe, 'Ca =', Ca, 'lamb =', lamb, 'theta1 =', theta1, 'theta2=', theta2)
    print('alpha =', alpha, 'nr =', nr, 'nz =', nz, 'Dt =', Dt, 'nr_steps =', nr_steps)
    print('dp =', dp)
    CHproblem = Problem(Pe, Cn, Ca, lamb, theta1, theta2, dp)
    nonrefinedmesh = read_capillary_and_make_mesh(n=nr, zmin=-10, zmax=7.5)

    CHproblem.load_mesh(nonrefinedmesh)
    CHproblem.refine_mesh(0, 0.95)
    CHproblem.create_function_space_mesh()
    x_new, f = emulsion_shape_init()
    CHproblem.set_x_new_f(x_new, f)
    CHproblem.initialize_field()
    CHproblem.split_functions()
    CHproblem.formulate_problem(Dt)
    CHproblem.refine_mesh(3, 0.95)
    CHproblem.create_function_space_mesh()
    CHproblem.initialize_field()
    CHproblem.split_functions()
    CHproblem.formulate_problem(Dt)
    t = 0
    fileu = File(file_name + 'u.pvd')
    filep = File(file_name + 'p.pvd')
    filephi = File(file_name + 'phi.pvd')
    filemu = File(file_name + 'mu.pvd')
    # fileu << (CHproblem.Ssol.split()[0], float(t))
    # filep << (CHproblem.Ssol.split()[1], float(t))
    # filephi << (CHproblem.CHsol.split()[0], float(t))
    # filemu << (CHproblem.CHsol.split()[1], float(t))
    R_list = []
    for i in (range(nr_steps)):

        print(i, '/', nr_steps - 1, 'time:', time.time() - start_time)
        t += Dt
        CHproblem.CHsol0.assign(CHproblem.CHsol)
        CHproblem.Ssol0.assign(CHproblem.Ssol)

        print('Solving CH')
        solve(CHproblem.CHL == 0, CHproblem.CHsol,
              solver_parameters={"newton_solver": {"linear_solver": "mumps"}})
        print('Solving S')
        solve(CHproblem.SL == 0, CHproblem.Ssol, CHproblem.BCs,
              solver_parameters={"newton_solver": {"linear_solver": "mumps"}})

        print('storing data')
        phi_value, mu_value = CHproblem.CHsol.split()
        u_value, p_value = CHproblem.Ssol.split()

        z_back = np.linspace(-10, 7.5, 100)
        for z_ba in z_back:
            if phi_value(0, z_ba) < 0:
                z_back_pos = z_ba
                break
            else:
                z_back_pos = 0


        R_dot_mu_1 = assemble((Ca * CHproblem.eta) * ((2
                            * Dx(CHproblem.u[0], 0)**2) + Dx(CHproblem.u[0], 1)
                            * (Dx(CHproblem.u[0], 1) + Dx(CHproblem.u[1], 0))
                            + 2 * (CHproblem.u[0] / CHproblem.r)**2
                            + Dx(CHproblem.u[1], 0) * (Dx(CHproblem.u[0], 1)
                            + Dx(CHproblem.u[1], 0)) + 2
                            * Dx(CHproblem.u[1], 1)**2) * dx)
        R_dot_D = assemble(Cn / Pe * ((CHproblem.mu - CHproblem.mu0) / Dt) **2 * dx)
        R_dot_rho = assemble(Ca / 2 * (CHproblem.u**2 - CHproblem.u0**2) / Dt * dx)
        print('R_dot_mu_1', R_dot_mu_1, 'R_dot_D', R_dot_D, 'R_dot_rho', R_dot_rho, 'z_back_pos', z_back_pos)
        with open(file_name + 'dissipation.txt', 'a') as f:
            f.write(str(R_dot_mu_1) + ' ' + str(R_dot_D) + ' ' + str(R_dot_rho) + ' ' + str(t) + ' ' + str(z_back_pos) + '\n')

        coords = CHproblem.mesh.coordinates()
        shape = np.shape(coords)
        shape0 = shape[0]
        shape1 = shape[1]
        phi = np.zeros(shape0)
        mu = np.zeros(shape0)
        p = np.zeros(shape0)
        u = np.zeros(shape)

        for j in range(shape0):
            phi[j] = phi_value(coords[j, 0], coords[j, 1])
            mu[j] = mu_value(coords[j, 0], coords[j, 1])
            p[j] = p_value(coords[j, 0], coords[j, 1])
            u[j] = u_value(coords[j, 0], coords[j, 1])

        if i % 25 == 0:
            coords.dump(file_name + str(i) + 'coords.dat')
            phi.dump(file_name + str(i) + 'phi.dat')
            mu.dump(file_name + str(i) + 'mu.dat')
            p.dump(file_name + str(i) + 'p.dat')
            u.dump(file_name + str(i) + 'u.dat')
            File(file_name + 'Ssol_.xml') << CHproblem.Ssol
            File(file_name + 'CHsol_.xml') << CHproblem.CHsol
            File(file_name + 'mesh_.xml') << CHproblem.mesh


        print('data stored')
        CHproblem.refine_mesh(3, 0.95)
        CHproblem.create_function_space_mesh()
        CHproblem.interpolate_field()
        CHproblem.split_functions()
        CHproblem.formulate_problem(Dt)

        if write_to_file:
            fileu << (CHproblem.Ssol.split()[0], float(t))
            filep << (CHproblem.Ssol.split()[1], float(t))
            filephi << (CHproblem.CHsol.split()[0], float(t))
            filemu << (CHproblem.CHsol.split()[1], float(t))



    time_spent = time.time() - start_time
    print(time_spent, 's')
    return CHproblem.Ssol, CHproblem.CHsol, R_list


def find_mass_center(phi_value, mu_value, r_values,
                     z_values):
    s = 0
    s_count = 0
    for r in r_values:
        for z in z_values:
            if phi_value(r, z) < 0:
                s += np.array([0, z])
                s_count += 1
    if s_count > 1:
        R = s / s_count
        return R
    else:
        return 'Bubble out of frame'


if __name__ == "__main__":
    set_log_level(24)
    import sys
    nr = 400
    nz = 0
    Cn = 0.01
    Pe = 1 / (3 * 0.01)
    Ca = 1
    lamb = 1.0 / 82.4
    alpha = 0.9
    Dt = 0.1
    theta1 = -pi
    theta2 = -pi
    dp = 5.0
    T = 2500
    nr_steps = int(T / Dt)
    write_to_file = False
    folder = 'dp=%1.2f' % dp + 'Cn=%1.2f' % Cn + \
        'Pe=%1.2f' % Pe + 'n_domain=%d' % nr + \
        'Ca=%1.2f' % Ca + 'lamb=%1.4f' % lamb + 'theta1=%1.4f' % theta1 + \
        'theta2=%1.4f' % theta2 + 'Dt=%1.2f' % Dt + '/'
    droplet = 'p1_100_p2_7_60fps_extracted_droplet_2'
    file_name = droplet + '/' + folder

    mmass_center = []
    Ssol, CHsol, R_list = SCHsolve(Dt=Dt, nr=nr, nz=nz, Pe=Pe, Cn=Cn, Ca=Ca, theta1=theta1, theta2=theta2,
                           lamb=lamb, alpha=alpha, nr_steps=nr_steps, dp=dp,
                           write_to_file=write_to_file, file_name=file_name)
    File(file_name + 'Ssol_final' + '.xml') << Ssol
    File(file_name + 'CHsol_final' + '.xml') << CHsol
