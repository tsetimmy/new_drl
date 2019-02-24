import numpy as np

def _basis(X, random_matrix, bias, basis_dim, length_scale, signal_sd):
    x_omega_plus_bias = np.matmul(X, (1./length_scale)*random_matrix) + bias
    z = signal_sd * np.sqrt(2./basis_dim) * np.cos(x_omega_plus_bias)
    return z

class RegressionWrapper:
    def __init__(self, input_dim, basis_dim, length_scale=1., signal_sd=1., noise_sd=5e-4, prior_sd=1., rffm_seed=1, train_hp_iterations=2000, matern_param=np.inf):
        self.input_dim = input_dim
        self.basis_dim = basis_dim
        self.length_scale = length_scale
        self.signal_sd = signal_sd
        self.noise_sd = noise_sd
        self.prior_sd = prior_sd
        self.hyperparameters = np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd])
        self.c = 1e-6

        self.rffm_seed = rffm_seed
        self.train_hp_iterations = train_hp_iterations

        self._init_statistics()

        rng_state = np.random.get_state()
        np.random.seed(self.rffm_seed)

        self.random_matrix = np.random.normal(size=[self.input_dim, self.basis_dim])
        if matern_param != np.inf:
            df = 2. * (matern_param + .5)
            u = np.random.chisquare(df, size=[self.basis_dim,])
            self.random_matrix = self.random_matrix * np.sqrt(df / u)
        self.bias = np.random.uniform(low=0., high=2.*np.pi, size=[self.basis_dim])

        np.random.set_state(rng_state)

    def _init_statistics(self):
        self.XX = np.zeros([self.basis_dim, self.basis_dim])
        self.Xy = np.zeros([self.basis_dim, 1])

    def _update(self, X, y):
        assert len(X.shape) == 2
        assert len(y.shape) == 2
        assert X.shape[0] == y.shape[0]

        basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)
        self.XX += np.matmul(basis.T, basis)
        self.Xy += np.matmul(basis.T, y)

        self.Llower = scipy.linalg.cholesky((self.noise_sd/self.prior_sd)**2*np.eye(self.basis_dim) + self.XX, lower=True)

    def _train_hyperparameters(self, X, y):
        warnings.filterwarnings('error')
        '''
        import cma
        thetas = np.copy(np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd]))
        options = {'maxiter': 1000, 'verb_disp': 1, 'verb_log': 0}
        res = cma.fmin(self._log_marginal_likelihood, thetas, 2., args=(X, y), options=options)
        results = np.copy(res[0])
        '''

        thetas = np.copy(np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd]))
        options = {'maxiter': self.train_hp_iterations, 'disp': True}
        _res = minimize(self._log_marginal_likelihood, thetas, method='powell', args=(X, y), options=options)
        results = np.copy(_res.x)

        self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd = results
        self.length_scale = np.abs(self.length_scale)
        self.signal_sd = np.abs(self.signal_sd)
        #self.noise_sd = np.abs(self.noise_sd)
        self.noise_sd = np.sqrt(self.noise_sd**2 + self.c*self.prior_sd**2)
        self.prior_sd = np.abs(self.prior_sd)
        self.hyperparameters = np.array([self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd])
        print(self.length_scale, self.signal_sd, self.noise_sd, self.prior_sd)

    def _log_marginal_likelihood(self, thetas, X, y):
        try:
            length_scale, signal_sd, noise_sd, prior_sd = thetas

            noise_sd2 = np.sqrt(noise_sd**2 + self.c*prior_sd**2)

            basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, np.abs(length_scale), np.abs(signal_sd))
            N = len(basis.T)
            XX = np.matmul(basis.T, basis)
            Xy = np.matmul(basis.T, y)

            tmp0 = (noise_sd2/prior_sd)**2*np.eye(self.basis_dim) + XX
            #tmp = np.matmul(Xy.T, scipy.linalg.solve(tmp0.T, Xy, sym_pos=True))

            #cho_factor = scipy.linalg.cho_factor(tmp0)
            #tmp = np.matmul(scipy.linalg.cho_solve(cho_factor, Xy).T, Xy)
            Llower = scipy.linalg.cholesky(tmp0, lower=True)
            LinvXy = scipy.linalg.solve_triangular(Llower, Xy, lower=True)
            tmp = np.matmul(LinvXy.T, LinvXy)

            s, logdet = np.linalg.slogdet(np.eye(self.basis_dim) + (prior_sd/noise_sd2)**2*XX)
            if s != 1:
                print('logdet is <= 0. Returning 10e100.')
                return 10e100

            lml = .5*(-N*np.log(noise_sd2**2) - logdet + (-np.matmul(y.T, y)[0, 0] + tmp[0, 0])/noise_sd2**2)
            #loss = -lml + (length_scale**2 + signal_sd**2 + noise_sd_abs**2 + prior_sd**2)*1.5
            loss = -lml
            return loss
        except Exception as e:
            print('------------')
            print(e, 'Returning 10e100.')
            print('************')
            return 10e100

    def _reset_statistics(self, X, y):
        self._init_statistics()
        self._update(X, y)

    def _predict(self, X):
        basis = _basis(X, self.random_matrix, self.bias, self.basis_dim, self.length_scale, self.signal_sd)

        #TODO: fix this.
        predict_sigma = np.sum(np.square(scipy.linalg.solve_triangular(self.Llower, basis.T, lower=True)), axis=0) * self.noise_sd**2 + self.noise_sd**2
        predict_sigma = predict_sigma[..., np.newaxis]
        tmp0 = scipy.linalg.solve_triangular(self.Llower, basis.T, lower=True).T
        tmp1 = scipy.linalg.solve_triangular(self.Llower, self.Xy, lower=True)
        predict_mu = np.matmul(tmp0, tmp1)

        return predict_mu, predict_sigma

