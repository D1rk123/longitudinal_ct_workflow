from tomosipo.torch_support import to_autograd

class BackprojectionOperator:
    """Transpose of the Forward operator

    The idea of having a dedicated class for the backprojection
    operator, which just saves a link to the "real" operator has
    been shamelessly ripped from OpTomo.

    We have the following property:

    >>> import tomosipo as ts
    >>> vg = ts.volume(shape=10)
    >>> pg = ts.parallel(angles=10, shape=10)
    >>> A = ts.operator(vg, pg)
    >>> A.T is A.T.T.T
    True

    It is nice that we do not allocate a new object every time we use
    `A.T`. If we did, users might save the transpose in a separate
    variable for 'performance reasons', writing

    >>> A = ts.operator(vg, pg)
    >>> A_T = A.T

    This is a waste of time.
    """

    def __init__(
        self,
        parent,
    ):
        """Create a new tomographic operator"""
        super(BackprojectionOperator, self).__init__()
        self.parent = parent

    def __call__(self, projection, out=None):
        """Apply operator

        :param projection: `np.array` or `Data`
            An input projection. If a numpy array, the shape must match
            the operator geometry. If the input volume is an instance
            of `Data`, its geometry must match the operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A projection dataset on which the volume has been forward
            projected.
        :rtype: `Data`

        """
        return self.parent._bp(projection, out)

    def transpose(self):
        """Return forward projection operator"""
        return self.parent

    @property
    def T(self):
        """The transpose of the backprojection operator

        This property returns the transpose (forward projection) operator.
        """
        return self.transpose()

    @property
    def domain(self):
        """The domain (projection geometry) of the operator"""
        return self.parent.range

    @property
    def range(self):
        """The range (volume geometry) of the operator"""
        return self.parent.domain

    @property
    def domain_shape(self):
        """The expected shape of the input (projection) data"""
        return self.parent.range_shape

    @property
    def range_shape(self):
        """The expected shape of the output (volume) data"""
        return self.parent.domain_shape

class AutogradOperator():
    def __init__(
        self,
        operator
    ):
        if operator.additive == True:
            raise ValueError("Additive operators are not supported")
        
        self.operator = operator
        self._fp_op = to_autograd(operator)
        self._bp_op = to_autograd(operator.T)
        self._transpose = BackprojectionOperator(self)

    def _fp(self, volume, out=None):
        if out is None:
            return self._fp_op(volume)
        else:
            out[...] = self._fp_op(volume)
            return out
        
    def _bp(self, projection, out=None):
        if out is None:
            return self._bp_op(projection)
        else:
            out[...] = self._bp_op(projection)
            return out

    def __call__(self, volume, out=None):
        """Apply operator

        :param volume: `np.array` or `Data`
            An input volume. If a numpy array, the shape must match
            the operator geometry. If the input volume is an instance
            of `Data`, its geometry must match the operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A projection dataset on which the volume has been forward
            projected.
        :rtype: `Data`

        """
        return self._fp(volume, out)

    def transpose(self):
        """Return backprojection operator"""
        return self._transpose

    @property
    def T(self):
        """The transpose operator

        This property returns the transpose (backprojection) operator.
        """
        return self.transpose()

    @property
    def astra_compat_vg(self):
        return self.operator.astra_compat_vg
        
    @property
    def astra_compat_pg(self):
        return self.operator.astra_compat_pg

    @property
    def domain(self):
        """The domain (volume geometry) of the operator"""
        return self.operator.domain

    @property
    def range(self):
        """The range (projection geometry) of the operator"""
        return self.operator.range

    @property
    def domain_shape(self):
        """The expected shape of the input (volume) data"""
        return self.operator.domain_shape

    @property
    def range_shape(self):
        """The expected shape of the output (projection) data"""
        return self.operator.range_shape
