module weathergenmod
    ! This module includes subroutines to calculate daily maximum and minimum temperature and cloud cover fraction
    ! based on an annual timeseries of monthly values of these variables
    ! The weather generator is based on the WGEN model (Richardson, 1981) with extension to use monthly summary
    ! data from Geng et al., 1986, and Geng and Auburn 1986.
    ! Additional statistical relationships for both temperature, cloudiness and wind speed have been produced
    ! by P.S. Sommer and J.O. Kaplan using global weather station datasets (GHCN and global synoptic cloud
    ! reports).
    !
    ! Coded in 2007-2009 by Jed Kaplan and Joe Melton, ARVE Group, EPFL/UVic, jed.kaplan@unil.ch,
    ! 2011, Shawn Koppenhoefer, shawn.koppenhoefer@unil.ch
    ! 2016, Philipp Sommer, philipp.sommer@unil.ch

    use parametersmod, only : sp,dp,i4
    use randomdistmod, only : randomstate


    implicit none

    public  :: metvars_in
    public  :: metvars_out
    public  :: rmsmooth
    public  :: weathergen

    private :: daymetvars
    private :: meansd

    !-------------------------------

    type metvars_in
        ! Derived datatype for the monthly weather generator input

        real(sp) :: prec    ! monthly total precipitation amount (mm)
        real(sp) :: wetd    ! number of days in month with precipitation
        real(sp) :: wetf    ! fraction of days in month with precipitation

        real(sp) :: tmin    ! minumum temperture (C)
        real(sp) :: tmax    ! maximum temperture (C)
        real(sp) :: cldf    ! cloud fraction (0=clear sky, 1=overcast) (fraction)
        real(sp) :: wind    ! wind speed (m/s)

        logical, dimension(2)  :: pday    !precipitation status: true if the day was a rain day
        type(randomstate)      :: rndst   !state of the random number generator
        real(sp), dimension(4) :: resid   !previous day's weather residuals

    end type metvars_in

    type metvars_out
        ! Derived datatype for the daily weather generator output

        real(sp) :: prec    ! 24 hour total precipitation (mm)
        real(sp) :: tmin    ! 24 hour mean minimum temperature (degC)
        real(sp) :: tmax    ! 24 hour mean maximum temperature (degC)
        real(sp) :: cldf    ! 24 hour mean cloud cover fraction 0=clear sky, 1=overcast (fraction)
        real(sp) :: wind    ! wind speed (m s-1)

        logical, dimension(2)  :: pday    !precipitation state
        type(randomstate)      :: rndst   !state of the random number generator, 15 elements
        real(sp), dimension(4) :: resid   !previous day's weather residuals

    end type metvars_out

    type daymetvars
        ! Derived datatype for monthly climate variables

        real(sp) :: tmax_mn     ! maximum temperature monthly mean (degC)
        real(sp) :: tmin_mn     ! minimum temperature mothly mean (degC)
        real(sp) :: cldf_mn     ! mean cloud fraction (fraction)
        real(sp) :: wind_mn     ! wind speed

        real(sp) :: tmax_sd     ! standard deviation of corresponding variable above
        real(sp) :: tmin_sd     ! "
        real(sp) :: cldf_sd     ! "
        real(sp) :: wind_sd     ! "

    end type daymetvars

    ! -----------------------------------------------------------------------------
    ! ------------------- Defaults for the namelist parameters --------------------
    ! -----------------------------------------------------------------------------

    real(sp)                 :: thresh = 5.0 ! Threshold for transition from gamma to gp distribution

    logical :: thresh_pctl = .false. ! interpret the thresh as percentile
    ! coefficient to esimate the gamma scale parameter via
    ! g_scale = g_scale_coeff * mean_monthly_precip / number_of_wet_days
    ! following Geng et al., 1986
    real(sp)                 :: g_scale_coeff = 1.268022 ! coefficient to esimate the gamma scale parameter

    real(sp)                 :: gp_shape = 1.5 ! shape parameter for the Generalized Pareto distribution

    real(sp), dimension(4,4) :: A = reshape((/ & ! A matrix used for cross correlation following Richardson_1984 equation (4)
        0.913437, 0.032532,  -0.020658, 0.000573, &
        0.488761, 0.137304,  -0.07251,  -0.046058, &
        -0.00199, -0.04573,  0.591761,  0.026439, &
        0.010905, -0.044171, -0.018568, 0.666672 /), (/4, 4/))
    real(sp), dimension(4,4) :: B = reshape((/ & ! B matrix used for cross correlation following Richardson_1984 equation (4)
        0.361854, 0.0,       0.0,      0.0, &
        0.11441,  0.802987,  0.0,      0.0, &
        0.144862, -0.060622, 0.782791, 0.0, &
        0.080593, -0.015829, 0.066186, 0.736713/), (/4, 4/))

    ! transition probability correlations
    real(sp) :: p11_1 = 0.254877     ! intercept of p11 best line fit
    real(sp) :: p11_2 = 0.745123     ! slope of p11 best line fit
    real(sp) :: p101_1 = 0.0         ! intercept of p101 best line fit
    real(sp) :: p101_2 = 0.846326    ! slope of p101 best line fit
    real(sp) :: p001_1 = 0.0         ! intercept of p001 best line fit
    real(sp) :: p001_2 = 0.724019    ! slope of p001 best line fit

    ! temperature and cloud correlation parameters corresponding to wet or dry day
    ! minimum temperature regression results
    real(sp) :: tmin_w1 = 1.164653        ! intercept of best line fit of tmin on wet days (see :f:subr:`meansd`)
    real(sp) :: tmin_w2 = 0.955787        ! slope of best line fit of tmin on wet days (see :f:subr:`meansd`)
    real(sp) :: tmin_sd_w1 = 3.038786     ! intercept of best line fit of std. dev. of tmin on wet days (see :f:subr:`meansd`)
    real(sp) :: tmin_sd_w2 = -0.050466    ! slope of best line fit of std. dev. of tmin on wet days (see :f:subr:`meansd`)
    real(sp) :: tmin_d1 = -0.528308       ! intercept of best line fit of tmin on dry days (see :f:subr:`meansd`)
    real(sp) :: tmin_d2 = 1.020964        ! slope of best line fit of tmin on dry days (see :f:subr:`meansd`)
    real(sp) :: tmin_sd_d1 = 3.544901     ! intercept of best line fit of std. dev. of tmin on dry days (see :f:subr:`meansd`)
    real(sp) :: tmin_sd_d2 = -0.042829    ! slope of best line fit of tmin on dry days (see :f:subr:`meansd`)

    ! maximum temperature regression results
    real(sp) :: tmax_w1 = -0.586296       ! intercept of best line fit of tmax on wet days (see :f:subr:`meansd`)
    real(sp) :: tmax_w2 = 0.948669        ! slope of best line fit of tmax on wet days (see :f:subr:`meansd`)
    real(sp) :: tmax_sd_w1 = 3.91589      ! intercept of best line fit of std. dev. of tmax on wet days (see :f:subr:`meansd`)
    real(sp) :: tmax_sd_w2 = -0.029385    ! slope of best line fit of std. dev. of tmax on wet days (see :f:subr:`meansd`)
    real(sp) :: tmax_d1 = 0.386508        ! intercept of best line fit of tmax on dry days (see :f:subr:`meansd`)
    real(sp) :: tmax_d2 = 1.0061          ! slope of best line fit of tmax on dry days (see :f:subr:`meansd`)
    real(sp) :: tmax_sd_d1 = 4.470039     ! intercept of best line fit of std. dev. of tmax on dry days (see :f:subr:`meansd`)
    real(sp) :: tmax_sd_d2 = -0.038717    ! slope of best line fit of tmax on dry days (see :f:subr:`meansd`)

    ! cloud regression results
    real(sp) :: cldf_w = -0.738271    ! *a* parameter for cloud fit on wet days (see :f:subr:`meansd`)
    real(sp) :: cldf_d = 0.420534     ! *a* parameter for cloud fit on dry days (see :f:subr:`meansd`)
    real(sp) :: cldf_sd_w = 0.981917  ! *a* parameter for std. dev. of cloud fit on wet days (see :f:subr:`meansd`)
    real(sp) :: cldf_sd_d = 1.041732  ! *a* parameter for std. dev. of cloud fit on dry days (see :f:subr:`meansd`)

    ! wind regression results
    real(sp) :: wind_w1 = 0.0           ! intercept of best line fit of wind on wet days (see :f:subr:`meansd`)
    real(sp) :: wind_w2 = 1.092938      ! slope of best line fit of wind on wet days (see :f:subr:`meansd`)
    real(sp) :: wind_d1 = 0.0           ! intercept of best line fit of wind on dry days (see :f:subr:`meansd`)
    real(sp) :: wind_d2 = 0.945229      ! slope of best line fit of wind on wet days (see :f:subr:`meansd`)
    real(sp) :: wind_sd_w1 = 0.0        ! intercept of best line fit of std. dev. of wind on wet days (see :f:subr:`meansd`)
    real(sp) :: wind_sd_w2 = 0.440435   ! slope of best line fit of std. dev. of wind on wet days (see :f:subr:`meansd`)
    real(sp) :: wind_sd_d1 = 0.0        ! intercept of best line fit of std. dev. wind on dry days (see :f:subr:`meansd`)
    real(sp) :: wind_sd_d2 = 0.512037   ! slope of best line fit of std. dev. wind on dry days (see :f:subr:`meansd`)

    ! wind bias correction (Note: Default is no correction)
    ! parameters of the slope - unorm best fit line
    real(sp) :: wind_slope_bias_L = 0.0   ! maximum value of logistic function of wind bias correction
    real(sp) :: wind_slope_bias_k = 0.0   ! steepness of logistic function of wind bias correction
    real(sp) :: wind_slope_bias_x0 = 0.0  ! x-value of sigmoid's midpoint of logistic function of wind bias correction

    ! coefficients for the bias correction of minimum temperature
    ! (Note: Default is no correction)
    real(sp), dimension(6) :: tmin_bias_coeffs = 0.0  ! coefficients for the bias correction of minimum temperature
    ! min. and max range for bias correction (1st and 99th percentile)
    real(sp) :: tmin_bias_min = -2.3263478740, tmin_bias_max = 2.3263478740 ! min. and max range for bias correction

    ! -----------------------------------------------------------------------------
    ! ------------------- END. Defaults for the namelist parameters ---------------
    ! -----------------------------------------------------------------------------

    ! the following parameters are computed by the cloud_params subroutine
    real(sp) :: cldf_w1, &
                cldf_w2, &
                cldf_w3, &
                cldf_w4, &
                cldf_d1, &
                cldf_d2, &
                cldf_d3, &
                cldf_d4

    contains

    !------------------------------------------------------------------------------------------------------------

    subroutine init_weathergen(f_unit)
        ! initialize the weather generator and read in the parameters from the
        ! namelist
        integer, intent(in), optional:: f_unit
        integer :: f_unit2 = 101

        namelist /weathergen_ctl/ &
            ! distribution parameters
            thresh, thresh_pctl, g_scale_coeff, gp_shape, &
            ! cross correlation coefficients
            A, B, &
            ! transition parameters
            p11_1, p11_2, p101_1, p101_2, p001_1, p001_2, &
            ! correlation parameters for wet days
            tmin_w1, tmin_w2, tmax_w1, tmax_w2, cldf_w, tmin_sd_w1, tmin_sd_w2, &
            tmax_sd_w1, tmax_sd_w2, cldf_sd_w, wind_w1, wind_w2, wind_sd_w1, &
            wind_sd_w2, &
            ! correlation parameters for dry days
            tmin_d1, tmin_d2, tmax_d1, tmax_d2, cldf_d, tmin_sd_d1, tmin_sd_d2, &
            tmax_sd_d1, tmax_sd_d2, cldf_sd_d, wind_d1, wind_d2, wind_sd_d1, &
            wind_sd_d2, &
            ! wind bias correction (Note: Default is no correction)
            wind_slope_bias_L, wind_slope_bias_k, wind_slope_bias_x0, &
            ! min. temperature bias correction (Note: Default is no correction)
            tmin_bias_coeffs, tmin_bias_min, tmin_bias_max

        if (.not. present(f_unit)) then
            open(f_unit2, file='weathergen.nml', status='old')
        else
            rewind f_unit
            f_unit2 = f_unit
        endif
        read(f_unit2, weathergen_ctl)
        if (.not. present(f_unit)) close(f_unit2)
        ! calculate cloud parameters
        call calc_cloud_params

    end subroutine init_weathergen

    !------------------------------------------------------------------------------------------------------------

    subroutine weathergen(met_in,met_out)

        use parametersmod, only : sp,dp,i4,ndaymonth,tfreeze
        use randomdistmod, only : ranur,ran_normal,ran_gamma_gp,ran_gamma, &
                                  gamma_cdf, gamma_pdf, gamma_cdf_inv

        implicit none

        !---------------
        !arguments

        type(metvars_in),  intent(in)  :: met_in
        type(metvars_out), intent(out) :: met_out

        !---------------
        !local variables

        integer  :: i

        real(sp) :: pre    ! monthly total precipitation amount (mm)
        real(sp) :: wetd   ! number of days in month with precipitation (fraction)
        real(sp) :: wetf   ! fraction of days in month with precipitation (fraction)
        real(sp) :: tmn    ! minumum temperture (C)
        real(sp) :: tmx    ! maximum temperture (C)
        real(sp) :: cld    ! cloud fraction (0=clear sky, 1=overcast) (fraction)
        real(sp) :: wnd    ! wind (m/s)

        real(sp), pointer :: tmax_mn
        real(sp), pointer :: tmin_mn
        real(sp), pointer :: cldf_mn
        real(sp), pointer :: wind_mn
        real(sp), pointer :: tmax_sd
        real(sp), pointer :: tmin_sd
        real(sp), pointer :: cldf_sd
        real(sp), pointer :: wind_sd

        type(randomstate) :: rndst       ! integer state of the random number generator
        logical,  dimension(2) :: pday   ! element for yesterday and the day before yesterday
        real(sp), dimension(4) :: resid  ! previous day's weather residuals

        real(sp) :: trange

        real(sp) :: prec
        real(sp) :: tmin
        real(sp) :: tmax
        real(sp) :: cldf
        real(sp) :: wind

        real(sp) :: pbar     ! mean amount of precipitation per wet day (mm)
        real(sp) :: pwet     ! probability that today will be wet
        real(sp) :: u        ! uniformly distributed random number (0-1)

        real(sp) :: g_shape
        real(sp) :: g_scale
        real(sp) :: gp_scale
        real(sp) :: thresh2use

        ! bias correction
        real(sp) :: slopecorr      ! slope correction for wind
        real(sp) :: tmin_bias      ! intercept correction for tmin

        real(kind=8) :: cdf_thresh  ! gamma cdf at the threshold
        real(kind=8) :: pdf_thresh  ! gamma pdf at the threshold

        type(daymetvars), target :: dmetvars

        real(sp), dimension(4) :: unorm  ! vector of uniformly distributed random numbers (0-1)

        !---------------------------------------------------------
        !input

        pre   = met_in%prec
        wetd  = met_in%wetd
        wetf  = met_in%wetf
        tmn   = met_in%tmin
        tmx   = met_in%tmax
        cld   = met_in%cldf
        wnd  = met_in%wind
        rndst = met_in%rndst
        pday  = met_in%pday
        resid = met_in%resid

        !shorthand to mean and CV structure

        tmin_mn => dmetvars%tmin_mn
        tmax_mn => dmetvars%tmax_mn
        cldf_mn => dmetvars%cldf_mn
        wind_mn => dmetvars%wind_mn
        tmin_sd => dmetvars%tmin_sd
        tmax_sd => dmetvars%tmax_sd
        cldf_sd => dmetvars%cldf_sd
        wind_sd => dmetvars%wind_sd

        !---------------------------
        !1) Precipitation occurrence

        !if there is precipitation this month, calculate the precipitation state for today

        if (wetf > 0. .and. pre > 0.) then

            !calculate transitional probabilities for dry to wet and wet to wet days
            !Relationships from Geng & Auburn, 1986, Weather simulation models based on summaries of long-term data

            if (pday(1)) then !yesterday was raining, use p11

                pwet = p11_1 + p11_2 * wetf

            else if (pday(2)) then !yesterday was not raining but the day before yesterday was raining, use p101

                pwet = p101_1 + p101_2 * wetf

            else  !both yesterday and the day before were dry, use p001

                pwet = p001_1 + p001_2 * wetf

            end if

            ! -----
            ! determine the precipitation state of the current day using the Markov chain approach
            u = ranur(rndst)

            if (u <= pwet) then  !today is a rain day

                pday = eoshift(pday,-1,.true.)

            else  !today is dry

                pday = eoshift(pday,-1,.false.)

            end if

            !---------------------------
            !2) precipitation amount

            if (pday(1)) then  !today is a wet day, calculate the rain amount

                !calculate parameters for the distribution function of precipitation amount

                pbar = pre / wetd

                !if (pbar > pmin) then
                !  g_scale = -2.16 + 1.83 * pbar !original relationship from Geng 1986
                !else
                !  g_scale = pbar
                !end if

                g_scale = g_scale_coeff * pbar
                g_shape = pbar / g_scale

                if (thresh_pctl) then
                    thresh2use = gamma_cdf_inv(real(thresh, kind=8), real(g_shape, kind=8), real(g_scale, kind=8))
                else
                    thresh2use = thresh
                end if

                call gamma_cdf(real(thresh2use, kind=8), 0.0_dp, real(g_scale, kind=8), &
                               real(g_shape, kind=8), cdf_thresh)
                call gamma_pdf(real(thresh2use, kind=8), 0.0_dp, real(g_scale, kind=8), &
                               real(g_shape, kind=8), pdf_thresh)

                gp_scale = (1.0 - cdf_thresh)/ pdf_thresh

                do  !enforce positive precipitation

                    !today's precipitation

                    prec = ran_gamma_gp(rndst,.true.,g_shape,g_scale,thresh2use,gp_shape,gp_scale)

                    prec = roundto(prec,1)    !simulated precipitation should have no more precision than the input (0.1mm)

                    if (prec > 0.) exit

                end do

            else

                prec = 0.

            end if

        else

            pday = .false.
            prec = 0.

        end if

        !---------------------------

        !3) temperature min and max, cloud fraction

        !calculate a baseline mean and SD for today's weather dependent on precip status

        call meansd(pday(1),tmn,tmx,cld,wnd,  dmetvars)

        ! use random number generator for the normal distribution

        do i = 1,4
            call ran_normal(rndst,unorm(i))
        end do

        !calculate today's residuals for weather variables

        resid = matmul(A,resid) + matmul(B,unorm)  !Richardson 1981, eqn 5; WGEN tech report eqn. 3

        tmin = roundto(resid(1) * tmin_sd + tmin_mn,1)
        tmax = roundto(resid(2) * tmax_sd + tmax_mn,1)

        cldf = resid(3) * cldf_sd + cldf_mn

        wind = max(0.0, resid(4) * sqrt(max(0.0, wind_sd)) + sqrt(max(0.0, wind_mn)))

        ! ---- wind bias correction
        if (wind_slope_bias_L > 0) then
            slopecorr = wind_slope_bias_L / ( 1 + exp( - wind_slope_bias_k * ( &
                resid(4) - wind_slope_bias_x0)))

            wind = wind / slopecorr
        end if

        wind = roundto(wind * wind, 1)

        ! ----- tmin bias correction
        tmin_bias = sum(tmin_bias_coeffs(:) * ( &
            max(tmin_bias_min, min(tmin_bias_max, resid(1))) ** (/ 0, 1, 2, 3, 4, 5 /)))
        tmin = tmin - roundto(tmin_bias, 1)


        !---
        !add checks for invalid values here
        if (cldf>1) then
            cldf = 1.0
        elseif (cldf < 0.0) then
            cldf = 0.0
        end if

        if (wind<0) then
            wind = 0.0
        end if

        if (tmin+Tfreeze < 0.) then
            write(0,*)'Unphysical min. temperature with ', tmin, 'K from a monthly mean ', &
                tmin_mn, 'degC with bias correction ', tmin_bias, 'K for residual', resid(1)
            stop 1
        elseif (tmax+Tfreeze < 0.) then
            write(0,*)'Unphysical max. temperature with ', tmax, 'K from a monthly mean ', &
                tmax_mn, 'degC'
            stop 1
        end if

        !---

        met_out%prec  = prec
        met_out%tmin  = tmin
        met_out%tmax  = tmax
        met_out%cldf  = cldf
        met_out%wind  = wind
        met_out%pday  = pday
        met_out%rndst = rndst
        met_out%resid = resid

    10 format(2i4,l4,f8.3,8f9.2,3f9.5)

    end subroutine weathergen

    !------------------------------------------------------------------------------------------------------------

    subroutine meansd(pday,tmn,tmx,cld,wind,dm)
        ! Adjust the monthly means of temperature, cloud and wind corresponding to the wet/dry state
        !
        ! This routine makes the first approximation inside the weather generator to adjust the monthly
        ! mean according to the wet/dry state using the best fit lines from the parameterization.
        !
        ! Min. and max. temperature, as well as the wind speed, are calculated via
        !
        ! .. math::
        !
        !     x_{w/d} = x_{w/d1} + x_{w/d2} \cdot \bar{x}
        !
        ! Where :math:`x` stands either for the :math:`T_{min}, T_{max}, T_{min, sd}, T_{max, sd}, wind`
        ! or :math:`wind_{sd}`. :math:`w/d` stands for the wet dry state deterimined by `pday`.
        !
        ! The cloud fraction is calculated via
        !
        ! .. math::
        !
        !     c_{w/d} = \frac{-a_{w/d} - 1}{a_{w/d}^2 * \bar{c} - a_{w/d}^2 - a_{w/d}}  - \frac{1}{a_{w/d}}
        !
        ! and it's standard deviation via
        !
        ! .. math::
        !
        !     c_{sd, w/d} = a_{sd, w/d}^2 \cdot c_{w/d} \cdot (1 - c_{w/d})
        implicit none

        logical,          intent(in)  :: pday    ! precipitation status (mm/day)
        real(sp),         intent(in)  :: tmn     ! smooth interpolation of monthly minimum temperature (degC)
        real(sp),         intent(in)  :: tmx     ! smooth interpolation of monthly maximum temperature (degC)
        real(sp),         intent(in)  :: cld     ! fraction (0-1)
        real(sp),         intent(in)  :: wind    ! wind speed (m/s)
        type(daymetvars), intent(out) :: dm      ! the :f:type:`daymetvars` for the first daily approximation

        !local variables

        !---

        if (pday) then  !calculate mean and SD for a wet day

            dm%tmin_mn = tmin_w1 + tmin_w2 * tmn

            dm%tmax_mn = tmax_w1 + tmax_w2 * tmx

            dm%wind_mn = wind_w1 + wind_w2 * wind

            dm%cldf_mn = cldf_w1 / (cldf_w2 * cld + cldf_w3) + cldf_w4

            dm%tmin_sd = tmin_sd_w1 + tmin_sd_w2 * dm%tmin_mn

            dm%tmax_sd = tmax_sd_w1 + tmax_sd_w2 * dm%tmax_mn

            dm%wind_sd = wind_sd_w1 + wind_sd_w2 * dm%wind_mn

            dm%cldf_sd = cldf_sd_w * dm%cldf_mn * (1. - dm%cldf_mn)

        else  !dry day

            dm%tmin_mn = tmin_d1 + tmin_d2 * tmn

            dm%tmax_mn = tmax_d1 + tmax_d2 * tmx

            dm%wind_mn = wind_d1 + wind_d2 * wind

            dm%cldf_mn = cldf_d1 / (cldf_d2 * cld + cldf_d3) + cldf_d4

            dm%tmin_sd = tmin_sd_d1 + tmin_sd_d2 * dm%tmin_mn

            dm%tmax_sd = tmax_sd_d1 + tmax_sd_d2 * dm%tmax_mn

            dm%wind_sd = wind_sd_d1 + wind_sd_d2 * dm%wind_mn

            dm%cldf_sd = cldf_sd_d * dm%cldf_mn * (1. - dm%cldf_mn)

        end if

    end subroutine meansd

    !----------------------------------------------------------------------------------------------------------------

    subroutine rmsmooth(m,dmonth,bcond,r)
        ! Iterative, mean preserving method to smoothly interpolate mean data to pseudo-sub-timestep values
        ! From Rymes, M.D. and D.R. Myers, 2001. Solar Energy (71) 4, 225-231

        use parametersmod, only : sp,dp

        implicit none

        !arguments
        real(sp), dimension(:), intent(in)  :: m      ! vector of mean values at super-time step (e.g., monthly), minimum three values
        integer,  dimension(:), intent(in)  :: dmonth ! vector of number of intervals for the time step (e.g., days per month)
        real(sp), dimension(2), intent(in)  :: bcond  ! boundary conditions for the result vector (1=left side, 2=right side)
        real(sp), dimension(:), intent(out) :: r      ! result vector of values at chosen time step

        !parameters
        real(sp), parameter :: ot = 1. / 3

        !local variables
        integer :: n
        integer :: ni
        integer :: a
        integer :: b
        integer :: i
        integer :: j
        integer :: k
        integer :: l
        integer, dimension(size(r)) :: g
        real(sp) :: ck

        real(sp), dimension(2) :: bc

        !----------

        n  = size(m)
        ni = size(r)

        bc = bcond

        !initialize the result vector
        i = 1
        do a = 1,n
            j = i
            do b = 1,dmonth(a)
                r(i) = m(a)
                g(i) = j
                i = i + 1
            end do
        end do

        !iteratively smooth and correct the result to preserve the mean

        !iteration loop
        do i = 1,ni

            do j = 2,ni-1
                r(j) = ot * (r(j-1) + r(j) + r(j+1))   !Eqn. 1
            end do

            r(1)  = ot * (bc(1)   + r(1)  +  r(2))   !Eqns. 2
            r(ni) = ot * (r(ni-1) + r(ni) + bc(2))

            j = 1
            do k = 1,n                               !calculate one correction factor per super-timestep

                a = g(j)                               !index of the first timestep value of the super-timestep
                b = g(j) + dmonth(k) - 1               !index of the last timestep value of the super-timestep

                ck = sum(m(k) - r(a:b)) / ni           !Eqn. 4

                do l = 1,dmonth(k)                     !apply the correction to all timestep values in the super-timestep
                    r(j) = r(j) + ck
                    j = j + 1
                end do

                !correction for circular conditions when using climatology (do not use for transient simulations)
                bc(1) = r(ni)
                bc(2) = r(1)

            end do
        end do

    end subroutine rmsmooth

    !-----------------------------------------------------------------------

    real(sp) function roundto(val,precision)
        ! round a value to the given precision

        implicit none

        real(sp), intent(in) :: val  ! the input value
        integer,  intent(in) :: precision  ! the precision

        real(sp) :: scale

        !----

        scale = 10.**precision

        roundto = real(nint(val * scale)) / scale

    end function roundto

    !------------------------------------------------------------------------------------------------------------

    subroutine calc_cloud_params
        ! Calculate the parameters used for the first approximation in :f:subr:`meansd`
        !
        ! This subroutine uses :f:var:`cldf_w`, :f:var:`cldf_d`, :f:var:`cldf_sd_w` and
        ! :f:var:`cldf_sd_d` to calculate the necessary parameters for the adjustment of
        ! the monthly cloud fraction mean depending on the wet/dry state

        cldf_w1 = -cldf_w - 1.0
        cldf_w2 = cldf_w * cldf_w
        cldf_w3 = -(cldf_w * cldf_w) - cldf_w
        cldf_w4 = - 1.0/cldf_w
        cldf_sd_w = cldf_sd_w * cldf_sd_w

        cldf_d1 = -cldf_d - 1.0
        cldf_d2 = cldf_d * cldf_d
        cldf_d3 = -(cldf_d * cldf_d) - cldf_d
        cldf_d4 = - 1.0/cldf_d
        cldf_sd_d = cldf_sd_d * cldf_sd_d

    end subroutine calc_cloud_params

end module weathergenmod
