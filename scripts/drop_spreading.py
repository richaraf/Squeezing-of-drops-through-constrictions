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

    def eval(self, values, x):
        Cn = self.Cn
        r = sqrt(x[0] * x[0] + (x[1] - 1) * (x[1] - 1))
        values[0] = tanh((r - 1.0) / (sqrt(2) * Cn))

    def value_shape(self):
        return (2,)


class Problem:
    def __init__(self, Pe, Cn, Ca, lamb, theta1, theta2):
        self.Pe = Pe
        self.Cn = Cn
        self.Ca = Ca
        self.lamb = lamb
        self.theta1 = theta1
        self.theta2 = theta2

    def load_mesh(self, mesh):
        self.nonrefinedmesh = mesh
        self.mesh = mesh

    def refine_mesh(self, nr_refinements, first_case=False):
        phi_boundary = 0.95
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
        botwall_firstpart = CompiledSubDomain('near(x[0], 6.0) && on_boundary')
        z_axis = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
        firstwall = CompiledSubDomain('near(x[1], 0.0) && on_boundary')
        secondwall = CompiledSubDomain('near(x[1], 2.5 - x[0] / 3) && on_boundary')
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
        boundaries = self.boundaries
        # Define boundary conditions
        bcu_firstwall = DirichletBC(self.S.sub(0), Constant((0, 0)), boundaries, 1)
        bcp_botwall = DirichletBC(self.S.sub(1), Constant((0)), boundaries, 2)
        bcp_secondwall = DirichletBC(self.S.sub(1), Constant((0)), boundaries, 3)
        bcu = [bcu_firstwall]
        bcp = [bcp_secondwall]
        self.BCs = bcp + bcu
        self.phi = variable(self.phi)
        F = 0.25 * (self.phi**2 - 1)**2
        dFdphi = diff(F, self.phi)
        g = 0.5 + 0.75 * self.phi - 0.25 * self.phi**3
        dgdphi = diff(g, self.phi)
        eta = 0.5 * (1 + self.phi + self.lamb * (1 - self.phi))
        x = SpatialCoordinate(self.mesh)
        r = x[0]
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
        L -= Dt * self.Cn * cos(self.theta1) * self.v * r * dgdphi * self.ds(1) # first part wetting condition
        #L -= Dt * self.Cn * cos(self.theta2) * self.v * r * dgdphi * self.ds(5) # last part wetting condition
        self.CHL = L


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


def SCHsolve(nr=50, nz=100, Pe=3e-3, Cn=0.06, Ca=1e0, theta1=0, theta2=0, lamb=1.0,
             alpha=1.0, nr_steps=25, write_to_file=False, file_name='', Dt=0.1):
    start_time = time.time()
    print('Cn =', Cn, 'Pe =', Pe, 'Ca =', Ca, 'lamb =', lamb, 'theta1 =', theta1, 'theta2=', theta2)
    print('alpha =', alpha, 'nr =', nr, 'nz =', nz, 'Dt =', Dt, 'nr_steps =', nr_steps)
    CHproblem = Problem(Pe, Cn, Ca, lamb, theta1, theta2)
    domain_vertices = [Point(0.0, 0.0),
                       Point(6.0, 0.0),
                       Point(6.0, 0.5),
                       Point(0.0, 2.5)]
    domain = Polygon(domain_vertices)
    nonrefinedmesh = generate_mesh(domain, nr)
    CHproblem.load_mesh(nonrefinedmesh)
    CHproblem.refine_mesh(0)
    CHproblem.create_function_space_mesh()
    CHproblem.initialize_field()
    CHproblem.split_functions()
    CHproblem.formulate_problem(Dt)

    t = 0
    Dt = 0.001
    CHproblem.refine_mesh(3)
    CHproblem.create_function_space_mesh()
    CHproblem.interpolate_field()
    CHproblem.split_functions()
    CHproblem.formulate_problem(Dt)
    if write_to_file:
        fileu = File(file_name + 'u.pvd')
        filep = File(file_name + 'p.pvd')
        filephi = File(file_name + 'phi.pvd')
        filemu = File(file_name + 'mu.pvd')
        fileu << (CHproblem.Ssol.split()[0], float(t))
        filep << (CHproblem.Ssol.split()[1], float(t))
        filephi << (CHproblem.CHsol.split()[0], float(t))
        filemu << (CHproblem.CHsol.split()[1], float(t))
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

        if i == 49:
            Dt = 0.0015
        if i == 99:
            Dt = 0.0030
        if i == 149:
            Dt = 0.01
        if i == 399:
            Dt = 0.05

        CHproblem.refine_mesh(3)
        CHproblem.create_function_space_mesh()
        CHproblem.interpolate_field()
        CHproblem.split_functions()
        CHproblem.formulate_problem(Dt)

        if write_to_file:# and (i == 49 or i == 149 or i == 239 or i == 399):
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
    set_log_level(30)
    import sys
    nr = 100
    nz = 2 * nr
    Cn = 0.01
    Pe = 1 / (3 * 0.01)
    Ca = 0.01
    lamb = 50.0
    alpha = 0.9
    Dt = 0.05
    theta1 = -pi / 6
    theta2 = -pi
    T = 10000
    nr_steps = int(T / Dt)
    write_to_file = False
    folder = 'Cn=%1.2f' % Cn + \
        'Pe=%1.2f' % Pe + 'n_domain=%d' % nr + \
        'Ca=%1.2f' % Ca + 'lamb=%1.4f' % lamb + 'theta1=%1.4f' % theta1 + \
        'theta2=%1.4f' % theta2 + 'Dt=%1.2f' % Dt + '/'
    file_name = 'drop_spreading/' + folder

    mmass_center = []
    Ssol, CHsol, R_list = SCHsolve(Dt=Dt, nr=nr, nz=nz, Pe=Pe, Cn=Cn, Ca=Ca, theta1=theta1, theta2=theta2,
                           lamb=lamb, alpha=alpha, nr_steps=nr_steps,
                           write_to_file=write_to_file, file_name=file_name)
