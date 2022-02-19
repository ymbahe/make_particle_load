def get_cosmology_params(name):
    """Get cosmology parameters for a named cosmology."""    
    cosmo = {}
    if name == 'Planck2013':
        cosmo['Omega0'] = 0.307
        cosmo['OmegaLambda'] = 0.693
        cosmo['OmegaBaryon'] = 0.04825
        cosmo['hubbleParam'] = 0.6777
        cosmo['sigma8'] = 0.8288
        cosmo['linear_powerspectrum_file'] = 'extended_planck_linear_powspec'
    elif name == 'Planck2018':
        cosmo['Omega0'] = 0.3111
        cosmo['OmegaLambda'] = 0.6889
        cosmo['OmegaBaryon'] = 0.04897
        cosmo['hubbleParam'] = 0.6766
        cosmo['sigma8'] = 0.8102
        cosmo['linear_powerspectrum_file'] = 'EAGLE_XL_powspec_18-07-2019.txt'
    else:
        raise ValueError(f"Invalid cosmology '{name}'!")

    cosmo['OmegaDM'] = cosmo['Omega0'] - cosmo['OmegaBaryon']

    return cosmo