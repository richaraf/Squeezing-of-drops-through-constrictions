from dolfin import *
from math import tanh
import time
import numpy as np
from mshr import *


class CH_InitialConditions1(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_Cn(self, Cn):
        self.Cn = Cn

    def eval(self, values, x):
        Cn = self.Cn
        r = sqrt((x[0] + 0) * (x[0] + 0) + (x[1] * 0.9 + 5.5)*(x[1] * 0.9 + 5.5))
        values[0] = tanh((1.05 * r - 1.0) / (sqrt(2) * Cn))

    def value_shape(self):
        return (2,)


class CH_InitialConditions2(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_Cn(self, Cn):
        self.Cn = Cn

    def eval(self, values, x):
        Cn = self.Cn
        r = sqrt((x[0] + 0) * (x[0] + 0) + (x[1] + 6.2) * (x[1] + 6.2))
        values[0] = tanh((1.5 * r - 1.0) / (sqrt(2) * 2 * Cn * 1.5))

    def value_shape(self):
        return (2,)


class Problem:
    def __init__(self, Pe, Cn, Ca1, Ca2, lamb, theta1, theta2, dp):
        self.Pe = Pe
        self.Cn = Cn
        self.Ca1 = Ca1
        self.Ca2 = Ca2
        self.lamb1 = lamb
        self.lamb2 = lamb
        self.theta1 = theta1
        self.theta2 = theta2
        self.dp = dp

    def load_mesh(self, mesh):
        self.nonrefinedmesh = mesh
        self.mesh = mesh

    def refine_mesh(self, nr_refinements, phi_boundary, first_case=False):
        mesh = self.nonrefinedmesh
        for i in range(nr_refinements):
            cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
            cell_markers.set_all(False)
            if not first_case:
                phi_value1, mu_value1 = self.CHsol1.split()
                phi_value2, mu_value2 = self.CHsol2.split()
                for c in cells(mesh):
                    coordinates = c.get_vertex_coordinates()
                    r_value = (coordinates[0])
                    z_value = (coordinates[1])
                    coord_move1 = self.Cn * 1
                    coord_move2 = self.Cn * 2
                    phi_value1_00 = abs(phi_value1(r_value, z_value))
                    phi_value2_00 = abs(phi_value2(r_value, z_value))
                    try:
                        phi_value1_01p = abs(phi_value1(r_value, z_value + coord_move1))
                        phi_value2_01p = abs(phi_value2(r_value, z_value + coord_move1))
                        phi_value1_01m = abs(phi_value1(r_value, z_value - coord_move1))
                        phi_value2_01m = abs(phi_value2(r_value, z_value - coord_move1))
                    except:
                        phi_value1_01p = abs(phi_value1(r_value, z_value))
                        phi_value2_01p = abs(phi_value2(r_value, z_value))
                        phi_value1_01m = abs(phi_value1(r_value, z_value))
                        phi_value2_01m = abs(phi_value2(r_value, z_value))
                    if phi_value1_00 < phi_boundary:
                        cell_markers[c] = True
                    elif phi_value1_01p < phi_boundary:
                        cell_markers[c] = True
                    elif phi_value1_01m < phi_boundary:
                        cell_markers[c] = True
                    elif phi_value2_00 < phi_boundary:
                        cell_markers[c] = True
                    elif phi_value2_01p < phi_boundary:
                        cell_markers[c] = True
                    elif phi_value2_01m < phi_boundary:
                        cell_markers[c] = True
                    else:
                        cell_markers[c] = False

            # Regine mesh
            mesh = refine(mesh, cell_markers)
        self.mesh = mesh
        boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1,
                                      value=0)
        botwall_secondpart = CompiledSubDomain('x[0] > 0.3 && on_boundary')
        botwall_firstpart = CompiledSubDomain('x[0] > 0.3 && on_boundary')
        z_axis = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
        firstwall = CompiledSubDomain('near(x[1], -9) && on_boundary')
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
        self.P5 = FiniteElement('Lagrange', self.mesh.ufl_cell(), degree=1)
        self.P6 = FiniteElement('Lagrange', self.mesh.ufl_cell(), degree=1)
        self.SF = MixedElement([self.P1, self.P2])
        self.CHF1 = MixedElement([self.P3, self.P4])
        self.CHF2 = MixedElement([self.P5, self.P6])
        self.S = FunctionSpace(self.mesh, self.SF)
        self.CH1 = FunctionSpace(self.mesh, self.CHF1)
        self.CH2 = FunctionSpace(self.mesh, self.CHF2)

        self.dSsol = TrialFunction(self.S)
        self.k, self.l = TestFunctions(self.S)

        self.dCHsol1 = TrialFunction(self.CH1)
        self.q1, self.v1 = TestFunctions(self.CH1)

        self.dCHsol2 = TrialFunction(self.CH2)
        self.q2, self.v2 = TestFunctions(self.CH2)

    def initialize_field(self):
        print('initializing')
        self.Ssol = Function(self.S)
        self.Ssol0 = Function(self.S)
        self.CHsol1 = Function(self.CH1)
        self.CHsol01 = Function(self.CH1)
        self.CHsol2 = Function(self.CH2)
        self.CHsol02 = Function(self.CH2)
        CHsol_init1 = CH_InitialConditions1(degree=1)
        CHsol_init1.set_Cn(self.Cn)
        self.CHsol1.interpolate(CHsol_init1)
        self.CHsol01.interpolate(CHsol_init1)
        CHsol_init2 = CH_InitialConditions2(degree=1)
        CHsol_init2.set_Cn(self.Cn)
        self.CHsol2.interpolate(CHsol_init2)
        self.CHsol02.interpolate(CHsol_init2)
        print('initializing done')

    def interpolate_field(self):
        print('interpolating')
        self.Ssol0 = interpolate(self.Ssol0, self.S)
        self.CHsol01 = interpolate(self.CHsol01, self.CH1)
        self.CHsol02 = interpolate(self.CHsol02, self.CH2)
        self.Ssol = interpolate(self.Ssol, self.S)
        self.CHsol1 = interpolate(self.CHsol1, self.CH1)
        self.CHsol2 = interpolate(self.CHsol2, self.CH2)
        print('interpolating done')

    def split_functions(self):
        self.u, self.p = split(self.Ssol)
        self.u0, self.p0 = split(self.Ssol0)
        self.phi1, self.mu1 = split(self.CHsol1)
        self.phi01, self.mu01 = split(self.CHsol01)
        self.phi2, self.mu2 = split(self.CHsol2)
        self.phi02, self.mu02 = split(self.CHsol02)

    def formulate_problem(self, Dt):
        bcu_botwall_firstpart = DirichletBC(self.S.sub(0), Constant((0, 0)), self.boundaries, 2)
        bcu_botwall_secondpart = DirichletBC(self.S.sub(0), Constant((0, 0)), self.boundaries, 5)
        bcp_firstwall = DirichletBC(self.S.sub(1), Constant((self.dp)), self.boundaries, 1)
        bcp_botwall_firstpart = DirichletBC(self.S.sub(1), Constant((0)), self.boundaries, 2)
        bcp_secondwall = DirichletBC(self.S.sub(1), Constant((0)), self.boundaries, 3)

        bcu = [bcu_botwall_firstpart]
        bcp = [bcp_secondwall, bcp_firstwall]
        self.BCs = bcu + bcp

        self.phi1 = variable(self.phi1)
        F1 = 0.25 * (self.phi1**2 - 1)**2
        dFdphi1 = diff(F1, self.phi1)
        g1 = 0.5 + 0.75 * self.phi1 - 0.25 * self.phi1**3
        dgdphi1 = diff(g1, self.phi1)
        eta1 = 0.5 * (1 + self.phi1 + self.lamb1 * (1 - self.phi1))
        self.phi2 = variable(self.phi2)
        F2 = 0.25 * (self.phi2**2 - 1)**2
        dFdphi2 = diff(F2, self.phi2)
        g2 = 0.5 + 0.75 * self.phi2 - 0.25 * self.phi2**3
        dgdphi2 = diff(g2, self.phi2)
        eta2 = 0.5 * (1 + self.phi2 + self.lamb2 * (1 - self.phi2))
        self.eta1 = eta1
        self.eta2 = eta2
        x = SpatialCoordinate(self.mesh)
        r = x[0]
        self.r = r

        ##################
        # Stokes equation:
        # r - component
        L0 = -Dx(self.p, 0) * self.k[0] * r**2 * dx
        L0 -= eta1 * Dx(r * self.u[0], 0) * (2 * self.k[0] + r * Dx(self.k[0], 0)) * dx
        L0 -= eta1 * Dx(self.u[0], 1) * Dx(self.k[0] * r**2, 1) * dx
        # z - component
        L1 = -Dx(self.p, 1) * r * self.k[1] * dx
        L1 -= eta1 * r * Dx(self.u[1], 0) * Dx(self.k[1], 0) * dx
        L1 -= eta1 * Dx(self.u[1], 1) * Dx(self.k[1] * r, 1) * dx
        # continuity equation
        L2 = Dx(r * self.u[0], 0) * self.l * dx
        L2 += Dx(self.u[1], 1) * self.l * r * dx
        L = L0 + L1 + L2
        # Coupling terms:
        L += self.mu1 / (self.Cn * self.Ca1) * Dx(self.phi1, 0) * self.k[0] * r**2 * dx
        L += self.mu1 / (self.Cn * self.Ca1) * Dx(self.phi1, 1) * self.k[1] * r * dx
        L += self.mu2 / (self.Cn * self.Ca2) * Dx(self.phi2, 0) * self.k[0] * r**2 * dx
        L += self.mu2 / (self.Cn * self.Ca2) * Dx(self.phi2, 1) * self.k[1] * r * dx
        self.SL = L
        # Cahn-Hillard:
        L = (self.phi1 - self.phi01) * self.q1 * r * dx
        L += Dt * self.u[0] * Dx(self.phi1, 0) * self.q1 * r * dx
        L += Dt * self.u[1] * Dx(self.phi1, 1) * self.q1 * r * dx
        L += Dt * 1 / self.Pe * r * Dx(self.mu1, 0) * Dx(self.q1, 0) * dx
        L += Dt * 1 / self.Pe * Dx(self.mu1, 1) * r * Dx(self.q1, 1) * dx
        L += Dt * self.mu1 * self.v1 * r * dx
        L -= Dt * dFdphi1 * self.v1 * r * dx
        L -= Dt * self.Cn**2 * r * Dx(self.phi1, 0) * Dx(self.v1, 0) * dx
        L -= Dt * self.Cn**2 * Dx(self.phi1, 1) * r * Dx(self.v1, 1) * dx
        # Contact angle boundary condition:
        L -= Dt * self.Cn * cos(self.theta1) * self.v1 * r * dgdphi1 * self.ds(2) # first part wetting condition
        L -= Dt * self.Cn * cos(self.theta2) * self.v1 * r * dgdphi1 * self.ds(5) # last part wetting condition
        self.CHL1 = L

        L = (self.phi2 - self.phi02) * self.q2 * r * dx
        L += Dt * self.u[0] * Dx(self.phi2, 0) * self.q2 * r * dx
        L += Dt * self.u[1] * Dx(self.phi2, 1) * self.q2 * r * dx
        L += Dt * 1 / self.Pe * r * Dx(self.mu2, 0) * Dx(self.q2, 0) * dx
        L += Dt * 1 / self.Pe * Dx(self.mu2, 1) * r * Dx(self.q2, 1) * dx
        L += Dt * self.mu2 * self.v2 * r * dx
        L -= Dt * dFdphi2 * self.v2 * r * dx
        L -= Dt * self.Cn**2 * r * Dx(self.phi2, 0) * Dx(self.v2, 0) * dx
        L -= Dt * self.Cn**2 * Dx(self.phi2, 1) * r * Dx(self.v2, 1) * dx
        # Contact angle boundary condition:
        L -= Dt * self.Cn * cos(self.theta1) * self.v2 * r * dgdphi2 * self.ds(2) # first part wetting condition
        L -= Dt * self.Cn * cos(self.theta2) * self.v2 * r * dgdphi2 * self.ds(5) # last part wetting condition
        self.CHL2 = L


def f(z):
    constrict_len = 2.725
    constrict_width = 0.51
    if np.abs(z) > constrict_len:
        func = 1
    else:
        func = -np.cos(z * np.pi / constrict_len) * constrict_width / 2 - constrict_width / 2 + 1
    return func

def create_mesh_constrict(zmin, zmax, n, file_name):
    z = np.linspace(zmin, zmax, n)
    function = [f(z_) for z_ in z]
    np.save(file_name + 'botwall', np.asarray(function))

    bottom_wall = [Point(f(z_), z_) for z_ in z]
    top_wall = [Point(0, zmax), Point(0, 0), Point(0, zmin)]
    domain = Polygon(bottom_wall + top_wall)
    mesh = generate_mesh(domain, n)
    return mesh


def read_capillary_and_make_mesh(zmin=-9, zmax=20, n=200):
    boundary_file = 'constriction_extract_coords.txt'
    folder = ''

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


def SCHsolve(nr=50, nz=100, Pe=3e-3, Cn=0.06, Ca1=1e0, Ca2=1e0, theta1=0, theta2=0, lamb=1.0,
             nr_steps=25, write_to_file=False, file_name='', Dt=0.1, dp=15.0):
    start_time = time.time()
    print('Cn =', Cn, 'Pe =', Pe, 'Ca1 =', Ca1, 'Ca2 =', Ca2, 'lamb =', lamb, 'theta1 =', theta1, 'theta2=', theta2)
    print('nr =', nr, 'nz =', nz, 'Dt =', Dt, 'nr_steps =', nr_steps)
    print('dp =', dp)
    zmin = -9
    zmax = 7.5
    nonrefinedmesh = create_mesh_constrict(zmin, zmax, nr, file_name)

    CHproblem = Problem(Pe, Cn, Ca1, Ca2, lamb, theta1, theta2, dp)
    CHproblem.load_mesh(nonrefinedmesh)
    CHproblem.refine_mesh(0, 0.95)
    CHproblem.create_function_space_mesh()
    CHproblem.initialize_field()
    CHproblem.split_functions()
    CHproblem.formulate_problem(Dt)
    CHproblem.refine_mesh(3, 0.95, first_case=False)
    CHproblem.create_function_space_mesh()
    CHproblem.initialize_field()
    CHproblem.split_functions()
    CHproblem.formulate_problem(Dt)
    t = 0
    fileu = File(file_name + 'u.pvd')
    filep = File(file_name + 'p.pvd')
    filephi1 = File(file_name + 'phi.pvd')
    filemu1 = File(file_name + 'mu.pvd')
    filephi2 = File(file_name + 'phi2.pvd')
    filemu2 = File(file_name + 'mu.pvd')
    # fileu << (CHproblem.Ssol.split()[0], float(t))
    # filep << (CHproblem.Ssol.split()[1], float(t))
    # filephi1 << (CHproblem.CHsol1.split()[0], float(t))
    # filemu1 << (CHproblem.CHsol1.split()[1], float(t))
    # filephi2 << (CHproblem.CHsol2.split()[0], float(t))
    # filemu2 << (CHproblem.CHsol2.split()[1], float(t))
    R_list = []
    for i in (range(nr_steps)):
        print(i, '/', nr_steps - 1, 'time:', time.time() - start_time)
        t += Dt
        CHproblem.CHsol01.assign(CHproblem.CHsol1)
        CHproblem.CHsol02.assign(CHproblem.CHsol2)
        CHproblem.Ssol0.assign(CHproblem.Ssol)

        print('Solving CH1')
        time_spent = time.time() - start_time
        print(time_spent, 's')
        solve(CHproblem.CHL1 == 0, CHproblem.CHsol1,
              solver_parameters={"newton_solver": {"linear_solver": "mumps"}})
        print('Solving CH2')
        time_spent = time.time() - start_time
        print(time_spent, 's')
        solve(CHproblem.CHL2 == 0, CHproblem.CHsol2,
              solver_parameters={"newton_solver": {"linear_solver": "mumps"}})
        print('Solving S')
        time_spent = time.time() - start_time
        print(time_spent, 's')
        solve(CHproblem.SL == 0, CHproblem.Ssol, CHproblem.BCs,
              solver_parameters={"newton_solver": {"linear_solver": "mumps"}})

        print('storing data')
        time_spent = time.time() - start_time
        print(time_spent, 's')
        phi_value1, mu_value1 = CHproblem.CHsol1.split()
        phi_value2, mu_value2 = CHproblem.CHsol2.split()
        u_value, p_value = CHproblem.Ssol.split()

        z_back = np.linspace(-9, 7.5, 100)
        for z_ba in z_back:
            if phi_value1(0, z_ba) < 0:
                z_back_pos = z_ba
                break
            else:
                z_back_pos = 0

        R_dot_mu = assemble((Ca1 * CHproblem.eta1) * ((2
                            * Dx(CHproblem.u[0], 0)**2) + Dx(CHproblem.u[0], 1)
                            * (Dx(CHproblem.u[0], 1) + Dx(CHproblem.u[1], 0))
                            + 2 * (CHproblem.u[0] / CHproblem.r)**2
                            + Dx(CHproblem.u[1], 0) * (Dx(CHproblem.u[0], 1)
                            + Dx(CHproblem.u[1], 0)) + 2
                            * Dx(CHproblem.u[1], 1)**2) * dx)
        R_dot_D1 = assemble(Cn / Pe * ((CHproblem.mu1 - CHproblem.mu01) / Dt) **2 * dx)
        R_dot_D2 = assemble(Cn / Pe * ((CHproblem.mu2 - CHproblem.mu02) / Dt) **2 * dx)
        R_dot_rho = assemble(Ca1 / 2 * (CHproblem.u**2 - CHproblem.u0**2) / Dt * dx)
        print('R_dot_mu', R_dot_mu, 'R_dot_D1', R_dot_D1, 'R_dot_D2', R_dot_D2, 'R_dot_rho', R_dot_rho, 'z_back_pos', z_back_pos)
        with open(file_name + 'dissipation.txt', 'a') as f:
            f.write(str(R_dot_mu) + ' ' + str(R_dot_D1) + ' ' + str(R_dot_D2) + ' ' + str(R_dot_rho) + ' ' + str(t) + ' ' + str(z_back_pos) + '\n')

        coords = CHproblem.mesh.coordinates()
        shape = np.shape(coords)
        shape0 = shape[0]
        shape1 = shape[1]
        phi = np.zeros(shape0)
        phi1 = np.zeros(shape0)
        mu1 = np.zeros(shape0)
        phi2 = np.zeros(shape0)
        mu2 = np.zeros(shape0)
        p = np.zeros(shape0)
        u = np.zeros(shape)

        for j in range(shape0):
            phi[j] = phi_value1(coords[j, 0], coords[j, 1])
            phi1[j] = phi_value1(coords[j, 0], coords[j, 1])
            mu1[j] = mu_value1(coords[j, 0], coords[j, 1])
            phi[j] += phi_value2(coords[j, 0], coords[j, 1])
            phi2[j] = phi_value2(coords[j, 0], coords[j, 1])
            mu2[j] = mu_value2(coords[j, 0], coords[j, 1])
            p[j] = p_value(coords[j, 0], coords[j, 1])
            u[j] = u_value(coords[j, 0], coords[j, 1])
        if i % 25 == 0:
            coords.dump(file_name + str(i) + 'coords.dat')
            phi.dump(file_name + str(i) + 'phi.dat')
            phi1.dump(file_name + str(i) + 'phi1.dat')
            mu1.dump(file_name + str(i) + 'mu1.dat')
            phi2.dump(file_name + str(i) + 'phi2.dat')
            mu2.dump(file_name + str(i) + 'mu2.dat')
            p.dump(file_name + str(i) + 'p.dat')
            u.dump(file_name + str(i) + 'u.dat')
            print('data stored')
        if i % 500 == 0:
            File(file_name + 'Ssol_' + str(i) + '.xml') << CHproblem.Ssol
            File(file_name + 'CHsol1_' + str(i) + '.xml') << CHproblem.CHsol1
            File(file_name + 'CHsol2_' + str(i) + '.xml') << CHproblem.CHsol2
            File(file_name + 'mesh_' + str(i) + '.xml') << CHproblem.mesh
        time_spent = time.time() - start_time
        print(time_spent, 's')
        CHproblem.refine_mesh(3, 0.95)
        CHproblem.create_function_space_mesh()
        CHproblem.interpolate_field()
        CHproblem.split_functions()
        CHproblem.formulate_problem(Dt)

        if write_to_file:
            fileu << (CHproblem.Ssol.split()[0], float(t))
            filep << (CHproblem.Ssol.split()[1], float(t))
            filephi1 << (CHproblem.CHsol1.split()[0], float(t))
            filemu1 << (CHproblem.CHsol1.split()[1], float(t))
            filephi2 << (CHproblem.CHsol2.split()[0], float(t))
            filemu2 << (CHproblem.CHsol2.split()[1], float(t))

    time_spent = time.time() - start_time
    print(time_spent, 's')
    return CHproblem.Ssol, CHproblem.CHsol, R_list

if __name__ == "__main__":
    set_log_level(30)
    import sys
    nr = 400
    nz = 0
    Cn = 0.01
    dp = 10.0
    Pe = 1 / (3 * 0.01)
    Ca1 = 5.0
    Ca2 = 1.0
    lamb = 1.0
    Dt = 0.10
    theta1 = -pi
    theta2 = -pi
    T = 2500
    nr_steps = int(T / Dt)
    write_to_file = False
    folder = 'dp=%1.2f' % dp + 'Cn=%1.2f' % Cn + \
        'Pe=%1.2f' % Pe + 'n_domain=%d' % nr + \
        'Ca1=%1.2f' % Ca1 + 'Ca2=%1.2f' % Ca2 + 'lamb=%1.4f' % lamb + 'theta1=%1.4f' % theta1 + \
        'theta2=%1.4f' % theta2 + 'Dt=%1.2f' % Dt + '/'
    file_name = 'three_phase/cos_mesh/' + folder
    Ssol, CHsol, R_list = SCHsolve(Dt=Dt, nr=nr, nz=nz, Pe=Pe, Cn=Cn, Ca1=Ca1, Ca2=Ca2, theta1=theta1, theta2=theta2,
                           lamb=lamb, nr_steps=nr_steps, dp=dp,
                           write_to_file=write_to_file, file_name=file_name)
