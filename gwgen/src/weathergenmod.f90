module weathergenmod

!This module includes subroutines to calculate daily maximum and minimum temperature and cloud cover fraction
!based on an annual timeseries of monthly values of these variables
!The weather generator is based on the WGEN model (Richardson, 1981) with extension to use monthly summary
!data from Geng et al., 1986, and Geng and Auburn 1986.
!Additional statistical relationships for both temperature and cloudiness have been produced
!by J.O. Kaplan using global weather station datasets (GSOD and global synoptic cloud reports).

!Coded in 2007-2009 by Jed Kaplan and Joe Melton, ARVE Group, EPFL/UVic, jed.kaplan@epfl.ch
!Shawn Koppenhoefer 2011

use parametersmod, only : sp,dp,i4
use randomdistmod, only : randomstate


implicit none

public  :: metvars_in
public  :: metvars_out
public  :: rmsmooth
public  :: daily
public  :: weathergen

private :: daymetvars
private :: meansd
private :: esat

!-------------------------------

type metvars_in

  real(sp) :: prec    !monthly total precipitation amount (mm)
  real(sp) :: wetd    !number of days in month with precipitation
  real(sp) :: wetf    !fraction of days in month with precipitation

  real(sp) :: tmin    !minumum temperture (C)
  real(sp) :: tmax    !maximum temperture (C)
  real(sp) :: cldf    !cloud fraction (0=clear sky, 1=overcast) (fraction)
  real(sp) :: wind    !wind speed (m/s)

  real(sp)               :: NI      !previous day's Nesterov index (degC 2)
  logical, dimension(2)  :: pday    !precipitation status: true if the day was a rain day
  type(randomstate)      :: rndst   !state of the random number generator
  real(sp), dimension(4) :: resid   !previous day's weather residuals

end type metvars_in

type metvars_out

  real(sp) :: prec    !24 hour total precipitation (mm)
  real(sp) :: tmin    !24 hour minimum temperature (C)
  real(sp) :: tmax    !24 hour maximum temperature (C)
  real(sp) :: tdew    !dewpoint temperature (C)
  real(sp) :: cldf    !24 hour mean cloud cover fraction 0=clear sky, 1=overcast (fraction)
  real(sp) :: lght    !lightning flashes (not sure about units)
  real(sp) :: wind    !wind speed (m s-1)
  real(sp) :: dayl    !daylength (h)
  real(sp) :: srad    !downwelling surface shortwave radiation (J m-2 d-1)
  real(sp) :: dpet    !total potential evapotranspiration (mm)
  real(sp) :: NI      !Nesterov index (degC 2)

  logical, dimension(2)  :: pday    !precipitation state
  type(randomstate)      :: rndst   !state of the random number generator, 15 elements
  real(sp), dimension(4) :: resid   !previous day's weather residuals

end type metvars_out

type daymetvars

  real(sp) :: tmax_mn     !maximum temperature monthly mean (degC)
  real(sp) :: tmin_mn     !minimum temperature mothly mean (degC)
  real(sp) :: cldf_mn     !mean cloud fraction (fraction)
  real(sp) :: wind_mn     !wind speed

  real(sp) :: tmax_sd     !standard deviation of corresponding variable above
  real(sp) :: tmin_sd     ! "
  real(sp) :: cldf_sd     ! "
  real(sp) :: wind_sd     ! "

end type daymetvars

LOGICAL :: ltested = .FALSE.

! -----------------------------------------------------------------------------
! ------------------- Defaults for the namelist parameters --------------------
! -----------------------------------------------------------------------------

! Threshold for transition from gamma to gp distribution
real(sp)                 :: thresh = 15.0
! coefficient to esimate the gamma scale parameter via
! g_scale = g_scale_coeff * mean_monthly_precip / number_of_wet_days
! following Geng et al., 1986
real(sp)                 :: g_scale_coeff = 1.24896
! shape parameter for the Generalized Pareto distribution
real(sp)                 :: gp_shape = 0.08303
! solved A and B matrices following Richardson_1984 equation (4)
real(sp), dimension(4,4) :: A = reshape((/ &
    0.915395, 0.032931, -0.020349, 4.3e-05, &
    0.500123, 0.138388, -0.072738, -0.047751, &
    0.002037, -0.047249, 0.589393, 0.027476, &
    0.009912, -0.0451, -0.018483, 0.673857 /), (/4, 4/))
real(sp), dimension(4,4) :: B = reshape((/ &
    0.35534,  0.0,       0.0,      0.0, &
    0.115141, 0.792894,  0.0,      0.0, &
    0.148959, -0.06072,  0.783592, 0.0, &
    0.080738, -0.016318, 0.065818, 0.729933/), (/4, 4/))

! transition probability correlations
real(sp) :: p11_1 = 0.2448, &
            p11_2 = 0.7552, &
            p101_1 = 0.0, &
            p101_2 = 0.8649, &
            p001_1 = 0.0, &
            p001_2 = 0.7000

! temperature and cloud correlation parameters corresponding to wet or dry day
! minimum temperature regression results
real(sp) :: tmin_w1 = 1.0388, &
            tmin_w2 = 0.9633, &
            tmin_sd_w1 = 3.6244, &
            tmin_sd_w2 = -0.0978, &
            tmin_d1 = -0.5473, &
            tmin_d2 = 1.0253, &
            tmin_sd_d1 = 4.1502, &
            tmin_sd_d2 = -0.0959

! maximum temperature regression results
real(sp) :: tmax_w1 = -0.0909, &
            tmax_w2 = 0.9316, &
            tmax_sd_w1 = 4.2203, &
            tmax_sd_w2 = -0.0497, &
            tmax_d1 = -0.1278, &
            tmax_d2 = 1.0295, &
            tmax_sd_d1 = 5.0683, &
            tmax_sd_d2 = -0.0702

! cloud regression results
real(sp) :: cldf_w = -0.7085, &
            cldf_d = 0.4555, &
            cldf_sd_w = 1.0310, &
            cldf_sd_d = 1.0696

! wind regression results
real(sp) :: wind_w1 = -0.03854, &
            wind_w2 = 1.05564, &
            wind_d1 = 0.04884, &
            wind_d2 = 0.96591, &
            wind_sd_w1 = 0.44369, &
            wind_sd_w2 = 0.29040, &
            wind_sd_d1 = 0.48558, &
            wind_sd_d2 = 0.32268

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
    thresh, g_scale_coeff, gp_shape, &
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
    wind_sd_d2

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
                          gamma_cdf, gamma_pdf

implicit none

!---------------
!arguments

type(metvars_in),  intent(in)  :: met_in
type(metvars_out), intent(out) :: met_out

!---------------
!parameters

real(sp), parameter :: pmin = 2.16d0 / 0.83d0 !minimum value for pbar when using Geng linear relationship,
                                              !below this value we use a 1:1 line, see below
real(sp), parameter :: small = 5.e-5

!Richardson M0 and M1 matrices calculated based on global station synthesis

!cross correlations and lag correlations provided for convenience

!cross correlations
!        tmax   tmin    cloud
! M0 = [ 1.000, 0.576, -0.078, &  !tmax
!        0.576, 1.000,  0.015, &  !tmin
!       -0.078, 0.015,  1.000  ]  !cloud

!lag 1-day correlations
!       tmax+1 tmin+1  cloud+1
!M1 = [ 0.420, 0.540, -0.088, &  !tmax
!       0.546, 0.883, -0.014, &  !tmin
!      -0.067,-0.096,  0.580  ]  !cloud

!solved A and B matrices following Richardson_1984 equation (4)
!precalculated using matrixmath.R


!real(sp), parameter, dimension(3,3) :: A = [ 0.150, 0.461, -0.062, &
!                                             0.045, 0.857, -0.015, &
!                                            -0.045, 0.003,  0.576  ]

!real(sp), parameter, dimension(3,3) :: B = [ 0.825, 0.000, 0.000, &
!                                             0.107, 0.455, 0.000, &
!                                            -0.027, 0.094, 0.808  ]

! real(sp), parameter, dimension(3,3) :: A = [ 0.150, 0.453, -0.061, &
!                                              0.042, 0.855, -0.015, &
!                                             -0.045, 0.003,  0.572  ]
!
! real(sp), parameter, dimension(3,3) :: B = [ 0.830, 0.000,  0.000, &
!                                              0.113, 0.462,  0.000, &
!                                             -0.027, 0.089,  0.811  ]

!---------------
!local variables

integer  :: i

real(sp) :: pre    !monthly total precipitation amount (mm)
real(sp) :: wetd   !number of days in month with precipitation (fraction)
real(sp) :: wetf   !fraction of days in month with precipitation (fraction)
real(sp) :: tmn    !minumum temperture (C)
real(sp) :: tmx    !maximum temperture (C)
real(sp) :: cld    !cloud fraction (0=clear sky, 1=overcast) (fraction)
real(sp) :: wnd   !wind (m/s)

real(sp), pointer :: tmax_mn
real(sp), pointer :: tmin_mn
real(sp), pointer :: cldf_mn
real(sp), pointer :: wind_mn
real(sp), pointer :: tmax_sd
real(sp), pointer :: tmin_sd
real(sp), pointer :: cldf_sd
real(sp), pointer :: wind_sd

type(randomstate) :: rndst       !integer state of the random number generator
logical,  dimension(2) :: pday   !element for yesterday and the day before yesterday
real(sp), dimension(4) :: resid  !previous day's weather residuals

real(sp) :: trange

real(sp) :: tmean  !rough approximation of daily mean temperature (max + min / 2)
real(sp) :: temp   !rough approximation of daily mean temperature (max + min / 2)
real(sp) :: prec
real(sp) :: tmin
real(sp) :: tmax
real(sp) :: cldf
real(sp) :: wind
real(sp) :: es
real(sp) :: tdew
real(sp) :: NI

real(sp) :: pbar     !mean amount of precipitation per wet day (mm)
real(sp) :: pwet     !probability that today will be wet
real(sp) :: u        !uniformly distributed random number (0-1)

real(sp) :: g_shape
real(sp) :: g_scale
real(sp) :: gp_scale

real(kind=8) :: cdf_thresh, pdf_thresh  ! gamma cdf and gamma pdf at the threshold

type(daymetvars), target :: dmetvars

real(sp), dimension(4) :: unorm             !vector of uniformly distributed random numbers (0-1)

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
NI    = met_in%NI

!shorthand to mean and CV structure

tmin_mn => dmetvars%tmin_mn
tmax_mn => dmetvars%tmax_mn
cldf_mn => dmetvars%cldf_mn
wind_mn => dmetvars%wind_mn
tmin_sd => dmetvars%tmin_sd
tmax_sd => dmetvars%tmax_sd
cldf_sd => dmetvars%cldf_sd
wind_sd => dmetvars%wind_sd

tmean  = 0.5 * (tmx + tmn)
trange = tmx - tmn

!---------------------------
!1) Precipitation occurrence

!if there is precipitation this month, calculate the precipitation state for today

if (wetf > 0. .and. pre > 0.) then

  !calculate transitional probabilities for dry to wet and wet to wet days
  !Relationships from Geng & Auburn, 1986, Weather simulation models based on summaries of long-term data

  if (pday(1)) then !yesterday was raining, use p11

    !pwet  =  0.227 + 0.790 * wetf                     !linear regression (r2 0.7889)
    !pwet  =  0.245 + 0.777 * wetf                     !linear regression (r2 0.848)
!    pwet = 0.239 + 0.781 * wetf                        !linear regression (r2 0.805)
    pwet = p11_1 + p11_2 * wetf

  !else

    !pwet = 0.00097 + 0.7225 * wetf  !p01, not used for now

  else if (pday(2)) then !yesterday was not raining but the day before yesterday was raining, use p101

    !pwet = 0.828 * wetf         !linear regression forced through origin (r2 XXX)
    !pwet = 0.834 * wetf         !linear regression forced through origin (r2 0.976)
!    pwet = 0.847 * wetf          !linear regression forced through origin (r2 0.962)
    pwet = p101_1 + p101_2 * wetf

  else  !both yesterday and the day before were dry, use p001

    !pwet = 0.704 * wetf         !linear regression forced through origin (r2 XXX)
    !pwet = 0.682 * wetf         !linear regression forced through origin (r2 0.971)
!    pwet = 0.694 * wetf          !linear regression forced through origin (r2 0.968)
    pwet = p001_1 + p001_2 * wetf


  end if

  !determine the precipitation state of the current day using the Markov chain approach

  u = ranur(rndst)

  !write(0,*)u,rndst%indx

  if (u <= pwet) then  !today is a rain day

    pday = eoshift(pday,-1,.true.)

  else  !today is dry

    pday = eoshift(pday,-1,.false.)

  end if

  !-----
  !write(0,*)pwet,u,pday

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

    call gamma_cdf(real(thresh, kind=8), 0.0_dp, real(g_scale, kind=8), &
                   real(g_shape, kind=8), cdf_thresh)
    call gamma_pdf(real(thresh, kind=8), 0.0_dp, real(g_scale, kind=8), &
                   real(g_shape, kind=8), pdf_thresh)

    gp_scale = (1.0 - cdf_thresh)/ pdf_thresh


    do  !enforce positive precipitation

      !today's precipitation

      prec = ran_gamma_gp(rndst,.true.,g_shape,g_scale,thresh,gp_shape,gp_scale)
!      prec = ran_gamma(rndst,.true.,g_shape,g_scale)

      prec = round(prec,1)    !simulated precipitation should have have more precision than the input (0.1mm)

      if (prec > 0.) exit

    end do

  else

    prec = 0.

  end if

else

  pday = .false.
  prec = 0.

end if

!-----

!3) temperature min and max, cloud fraction

!calculate a baseline mean and SD for today's weather dependent on precip status

call meansd(pday(1),tmn,tmx,cld,wnd,  dmetvars)

!use random number generator for the normal distribution

do i = 1,4
  call ran_normal(rndst,unorm(i))
end do

!calculate today's residuals for weather variables

resid = matmul(A,resid) + matmul(B,unorm)  !Richardson 1981, eqn 5; WGEN tech report eqn. 3

tmin = round(resid(1) * tmin_sd + tmin_mn,1)
tmax = round(resid(2) * tmax_sd + tmax_mn,1)

cldf = resid(3) * cldf_sd + cldf_mn

wind = (resid(4) * wind_sd**0.5 + wind_mn**0.5) ** 2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TEST WISE IMPLEMENTATION
!wind = wind / (1.0079 + 0.7226 * unorm(4))
wind = (wind - 0.68421791 ** (3.21182244 - 3.15247364 * unorm(4))) / & 
       (1.007898 + 0.722561 * unorm(4))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!PRINT *, resid(3),cldf_sd,cldf_mn, cldf

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

!---
!calculate dewpoint
!To estimate dewpoint temperature we use the day's minimum temperature
!this makes the asumption that there is a close correlation between Tmin and dewpoint
!see, e.g., Glassy & Running, Ecological Applications, 1994

if (tmin+Tfreeze < 0.) then
  write(0,*)tmn,tmin_mn,tmin
  stop
end if

es = 0.01 * esat(tmin+Tfreeze) !saturation vapor pressure (mbar)

tdew = 34.07 + 4157. / log(2.1718e8 / es) !Josey et al., Eqn. 10 (K)

!---
!convert calculated temperatures from K to degC

!tmin = tmin - Tfreeze
!tmax = tmax - Tfreeze
!tdew = tdew - Tfreeze

temp = 0.5 * (tmx + tmn)

!---
!Nesterov index, Thonicke et al. (2010) eqn. 5

if (prec <= 3. .and. temp > 0.) then
  NI = NI + tmx * (tmx - (tmn - 4.))
else
  NI = 0.
end if

!write(*,'(6f12.2)')wetf,pre/wetd,prec,tmx,tmn,NI

!Nesterov index, Venevsky et al. (2002) eqn. 3

!tmean = 0.5 * (tmx + tmn)

!if (prec <= 3. .and. tmean > 0.) then
!  NI = NI + tmean * (tmean - (tmn - 4.))
!else
!  NI = 0.
!end if

!write(*,'(6f12.2)')wetf,pre/wetd,prec,tmean,tmn,NI

!---

met_out%prec  = prec
met_out%tmin  = tmin
met_out%tmax  = tmax
met_out%tdew  = tdew
met_out%cldf  = cldf
met_out%wind  = wind
met_out%pday  = pday
met_out%rndst = rndst
met_out%resid = resid
met_out%NI    = NI

10 format(2i4,l4,f8.3,8f9.2,3f9.5)

end subroutine weathergen

!------------------------------------------------------------------------------------------------------------

subroutine meansd(pday,tmn,tmx,cld,wind,dm)             !APRIL2011

implicit none

!arguments

logical,          intent(in)  :: pday    !precipitation status
real(sp),         intent(in)  :: tmn     !smooth interpolation of monthly minimum temperature (degC)
real(sp),         intent(in)  :: tmx     !smooth interpolation of monthly maximum temperature (degC)
real(sp),         intent(in)  :: cld     !fraction (0-1)
real(sp),         intent(in)  :: wind    !wind speed
type(daymetvars), intent(out) :: dm

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

real(sp) function esat(temp)

  !Function to calculate saturation vapor pressure (Pa) in water and ice
  !From CLM formulation, table 5.2, after Flatau et al. 1992

  use parametersmod, only : tfreeze

  implicit none

  real(sp), intent(in) :: temp !temperature in K

  real(sp) :: T        !temperature (degC)

  real(sp), dimension(9)   :: al !coefficients for liquid water
  real(sp), dimension(9)   :: ai !coefficients for ice
  real(sp), dimension(0:8) :: a  !coefficients

  integer :: i

  !--------------

  al(1) = 6.11213476
  al(2) = 4.44007856e-1
  al(3) = 1.43064234e-2
  al(4) = 2.64461437e-4
  al(5) = 3.05903558e-6
  al(6) = 1.96237241e-8
  al(7) = 8.92344772e-11
  al(8) =-3.73208410e-13
  al(9) = 2.09339997e-16

  ai(1) = 6.11123516
  ai(2) = 5.03109514e-1
  ai(3) = 1.88369801e-2
  ai(4) = 4.20547422e-4
  ai(5) = 6.14396778e-6
  ai(6) = 6.02780717e-8
  ai(7) = 3.87940929e-10
  ai(8) = 1.49436277e-12
  ai(9) = 2.62655803e-15

  if (temp <= tfreeze) then   !these coefficients are for temperature values in Celcius
    a(0:8) = ai
  else
    a(0:8) = al
  end if

  T = temp - tfreeze

  esat = a(0)

  do i = 1,8
    esat = esat + a(i) * T**i
  end do

  esat = 100. * esat

end function esat

!------------------------------------------------------------------------------------------------------------

subroutine rmsmooth(m,dmonth,bcond,r)

!Iterative, mean preserving method to smoothly interpolate mean data to pseudo-sub-timestep values
!From Rymes, M.D. and D.R. Myers, 2001. Solar Energy (71) 4, 225-231

use parametersmod, only : sp,dp

implicit none

!arguments
real(sp), dimension(:), intent(in)  :: m      !vector of mean values at super-time step (e.g., monthly), minimum three values
integer,  dimension(:), intent(in)  :: dmonth !vector of number of intervals for the time step (e.g., days per month)
real(sp), dimension(2), intent(in)  :: bcond  !boundary conditions for the result vector (1=left side, 2=right side)
real(sp), dimension(:), intent(out) :: r      !result vector of values at chosen time step

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

!------------------------------------------------------------------------------------------------------------

subroutine daily(mval,dval,means)

!linear interpolation of monthly to pseudo-daily values

implicit none

integer, parameter, dimension(12) :: ndaymo = (/  31,28,31,30,31,30,31,31,30,31,30,31 /)
integer, parameter, dimension(14) :: midday = (/ -15,16,44,75,105,136,166,197,228,258,289,319,350,381 /) !middle day of each month

real,    intent(in),  dimension(12)  :: mval
logical, intent(in) :: means

real, intent(out), dimension(365) :: dval

real, dimension(14) :: emval
real, dimension(13) :: slope

integer :: day
integer :: d
integer :: m

!-------------------------------------------------------
!interpolate the monthly values to daily ones in a cyclical format

!copy last month's data to first and vice versa

emval(2:13) = mval
emval(1) = mval(12)
emval(14) = mval(1)

if (means) then

  !calculate slopes

  forall (m = 1:13)
    slope(m) = (emval(m+1) - emval(m)) / (midday(m+1) - midday(m))
  end forall

  !calculate daily values based on monthly means

  m = 1
  do day = 1,365
    if (day > midday(m+1)) m = m + 1
    d = day - midday(m+1)
    dval(day) = slope(m) * d + emval(m+1)
  end do

else

  !distribute the total evenly among the days of the month

  day = 1
  do m = 1,12
    do d = 1,ndaymo(m)
      dval(day) = mval(m) / ndaymo(m)
      day = day + 1
    end do
  end do

end if

end subroutine daily

!------------------------------------------------------------------------------------------------------------

real(sp) function round(val,precision)

implicit none

real(sp), intent(in) :: val
integer,  intent(in) :: precision

real(sp) :: scale

!---

scale = 10.**precision

round = real(nint(val * scale)) / scale

end function round

!------------------------------------------------------------------------------------------------------------

subroutine calc_cloud_params

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
