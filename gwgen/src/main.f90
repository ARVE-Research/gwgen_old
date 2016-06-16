program weathergen_precip

!very simple version of the weather generator just for precipitation

use parametersmod, only : sp
use weathergenmod, only : metvars_in, metvars_out, weathergen, init_weathergen, &
                          rmsmooth
use randomdistmod, only : ran_seed
use geohashmod,    only: geohash

implicit none

character(200) :: infile
character(200) :: output
character(200) :: dailyfile

integer, parameter :: n = 1
integer, parameter :: n_curr = n + 1
integer, parameter :: n_tot = n * 2 + 1
character(12), parameter :: NOSTATION = '__NOSTATION'

integer :: i, d, rmin, rd, lmin, ld, curr_start, curr_end, end_counter = n
integer :: year(n_tot) = 0     ! month of previous, current, next and the month after the next month
integer :: month(n_tot) = 0    ! year of previous, current, next and the month after the next month

character(12) :: stationid(n_tot) = NOSTATION  ! previous, current and next station

real(sp) :: lon(n_tot) = 0  ! longitude of the station of previous, current, next and the month after the next month
real(sp) :: lat(n_tot) = 0  ! latitude of the station of previous, current, next and the month after the next month

real(sp) :: mprec(n_tot) = 0    ! precipitation amount of current month
integer  :: mwet(n_tot) = 0      ! number of wet days of current month
real(sp) :: mtmin(n_tot) = 0  ! min temperture of previous, current, next and the month after the next month
real(sp) :: mtmax(n_tot) = 0  ! min temperture of previous, current, next and the month after the next month
real(sp) :: mcloud(n_tot) = 0 ! cloudiness of previous, current, next and the month after the next month

real(sp), target :: tmin_sm(n_tot * 31) = 0   ! smoothed daily values of min temperature
real(sp), target :: tmax_sm(n_tot * 31) = 0   ! smoothed daily values of max temperature
real(sp), target :: cloud_sm(n_tot * 31) = 0  ! smoothed daily values of cloudiness

real(sp) :: bcond_tmin(2) = 0      ! boundary conditions of min temp for smoothing
real(sp) :: bcond_tmax(2) = 0      ! boundary conditions of max temp for smoothing
real(sp) :: bcond_cloud(2) = 0     ! boundary conditions of cloud for smoothing

integer :: i_consecutives(n_tot - 1)

! pointers to the tmin, tmax and cloud values of the current month
real(sp), pointer :: mtmin_curr(:), mtmax_curr(:), mcloud_curr(:)

integer  :: mwetd_sim
real(sp) :: mprec_sim

integer :: pdaydiff
integer :: pdaydiff1 = 31

real(sp) :: precdiff
real(sp) :: precdiff1 = 1000.

type(metvars_in)  :: met_in
type(metvars_out) :: met_out

integer :: ndm(n_tot) = 0

real(sp) :: prec_t

integer :: i_count

type(metvars_out), dimension(31) :: month_met

! -----------------------------------------------------------------------------
! ------------------- Defaults for the namelist parameters --------------------
! -----------------------------------------------------------------------------
! seed for the random number generator. Has no effect if use_geohash is .true.
integer :: seed = -30000
! logical to determines whether the geohashmod module shall be used for
! defining the random seed. If .true., the first three columns must be
! stationid, longitude, latitude. Otherwise longitude and latitude should be
! skipped
logical :: use_geohash = .true.

! -----------------------------------------------------------------------------
! ------------------- END. Defaults for the namelist parameters ---------------
! -----------------------------------------------------------------------------

namelist / main_ctl / seed, use_geohash

open(101,file='weathergen.nml',status='old')
read(101, main_ctl)
call init_weathergen()
close(101)

!initialize random state


call getarg(1,infile)
call getarg(2,output)

open(10,file=infile,status='old')

! read in the first n months and calculate the wet days


if (use_geohash) then
  do i=n_curr,n_tot-1

    read(10,*)stationid(i),lon(i),lat(i),year(i), month(i),mtmin(i),mtmax(i), &
               mcloud(i),mprec(i),mwet(i)
    ndm(i) = ndaymonth(year(i), month(i))

  end do
else
  do i=n_curr,n_tot-1

    read(10,*)stationid(i),year(i), month(i),mtmin(i),mtmax(i), &
               mcloud(i),mprec(i),mwet(i)
    ndm(i) = ndaymonth(year(i), month(i))

  end do
  call ran_seed(seed,met_in%rndst)
endif

! open the output file
dailyfile = trim(output)//'_daily.txt'

open(30,file=dailyfile,status='unknown')

do  !read the input file until the end

  ! nullify pointers
  nullify(mtmin_curr)
  nullify(mtmax_curr)
  nullify(mcloud_curr)

  !initialize weather residuals and other variables that carry over from one day to the next
  !these and the random state below should be reset once per station

  met_out%NI    = 0.
  met_out%pday(1) = .false.
  met_out%pday(2) = .false.
  met_out%resid = 0.

  !read in one month of summary weather station data from a text file
  if (use_geohash) then
    read(10,*,end=99)stationid(n_tot),lon(n_tot),lat(n_tot),year(n_tot),month(n_tot), &
                   mtmin(n_tot),mtmax(n_tot),mcloud(n_tot),mprec(n_tot),mwet(n_tot)
    ndm(n_tot) = ndaymonth(year(n_tot),month(n_tot))
    if (stationid(n_curr) /= stationid(n)) then
        call ran_seed(geohash(lon(n_curr), lat(n)),met_in%rndst)
    end if
  else
    read(10,*,end=99)stationid(n_tot),lon(n_tot),lat(n_tot),year(n_tot),month(n_tot), &
                   mtmin(n_tot),mtmax(n_tot),mcloud(n_tot),mprec(n_tot),mwet(n_tot)
    ndm(n_tot) = ndaymonth(year(n_tot),month(n_tot))
  endif

  goto 110

  ! ------ this part is skipped if we are not already at the end of the file -------------
99 continue
    end_counter = end_counter - 1
    stationid(n_tot) = NOSTATION
  ! ---------------------------------------------------------------------------------------
110 continue

  i_consecutives(:) = are_consecutive_months(stationid,year,month)

  if (any(i_consecutives(n:1:-1) == 0)) then
    lmin = n_curr - minloc(i_consecutives(n:1:-1), 1) + 1
    ld = SUM(ndm(:lmin - 1))
    bcond_tmin(1) = mtmin(lmin - 1)
    bcond_tmax(1) = mtmax(lmin - 1)
    bcond_cloud(1) = mcloud(lmin - 1)
  else
    lmin = 1
    ld = 1
  end if

  if (any(i_consecutives(n+1:) == 0)) then
    rmin = n + minloc(i_consecutives(n+1:), 1)
    bcond_tmin(2) = mtmin(rmin)
    bcond_tmax(2) = mtmax(rmin)
    bcond_cloud(2) = mcloud(rmin)
  else
    rmin = n_tot
    bcond_tmin(2) = mtmin(n_tot)
    bcond_tmax(2) = mtmax(n_tot)
    bcond_cloud(2) = mcloud(n_tot)
  end if
  rd = SUM(ndm(:rmin))

  call rmsmooth(mtmin(lmin:rmin), ndm(lmin:rmin), bcond_tmin, tmin_sm(ld:rd))
  call rmsmooth(mtmax(lmin:rmin), ndm(lmin:rmin), bcond_tmax, tmax_sm(ld:rd))
  call rmsmooth(mcloud(lmin:rmin), ndm(lmin:rmin), bcond_cloud, cloud_sm(ld:rd))

  curr_start = sum(ndm(:n))
  curr_end = curr_start + ndm(n+1) - 1

  mtmin_curr => tmin_sm(curr_start:curr_end)
  mtmax_curr => tmax_sm(curr_start:curr_end)
  mcloud_curr => cloud_sm(curr_start:curr_end)

  met_in%prec = mprec(n_curr)
  met_in%wetd = real(mwet(n_curr))
  met_in%wetf = real(mwet(n_curr)) / real(ndm(n_curr))
  met_in%NI    = 0.   !dummy value

  !write(0,*)stationid,year,month,met_in%prec,met_in%wetd,met_in%wetf

  prec_t = max(0.5,0.05 * mprec(n_curr))  !set quality threshold for preciptation amount

  i_count = 1

  do

    mwetd_sim = 0
    mprec_sim = 0.

    !start day loop

    do d = 1,ndm(n_curr)

      met_in%tmin  = mtmin_curr(d)
      met_in%tmax  = mtmax_curr(d)
      met_in%cldf  = real(mcloud_curr(d))
      met_in%pday  = met_out%pday
      met_in%resid = met_out%resid

      call weathergen(met_in,met_out)

      met_in%rndst = met_out%rndst

      month_met(d) = met_out

      if (met_out%prec > 0.) then
        mwetd_sim = mwetd_sim + 1
        mprec_sim = mprec_sim + met_out%prec
      end if

    end do

    !end of month

    if (mprec(n_curr) == 0.) then

      pdaydiff = 0
      precdiff = 0.

      exit

    else if (i_count >= 2) then  !enforce at least two times over the month to get initial values ok

      pdaydiff = abs(mwet(n_curr) - mwetd_sim)
      precdiff = abs(mprec(n_curr) - mprec_sim)

      if (pdaydiff <= 1 .and. precdiff <= prec_t) then  !restrict simulated total monthly precip to +/-5% or 0.5 mm of observed value
        exit

      else if (pdaydiff < pdaydiff1 .and. precdiff < precdiff1) then
        !save the values you have in a buffer in case you have to leave the loop

        !month_best = month_met

        pdaydiff1 = pdaydiff
        precdiff1 = precdiff

      end if

    end if

    i_count = i_count + 1

  end do

  !write out final results for this station/year/month combo and

  do d = 1,ndm(n_curr)
    write(30,'(a,3i5,2f10.1,f10.4,f10.1)')stationid(n_curr),year(n_curr),month(n_curr),&
        d,month_met(d)%tmin,month_met(d)%tmax,month_met(d)%cldf,month_met(d)%prec

  end do

  ! set boundary conditions for next timestep
  ! left boundary condition: the end of the last month
  ! right boundary condition: we try to use the month after the next two months
  ! (see code above) but if that does not work, we use here the end of the
  ! month after the next month
  bcond_tmin(:) = (/ tmin_sm(ld), tmin_sm(rd) /)
  bcond_tmax(:) = (/ tmax_sm(ld), tmax_sm(rd) /)
  bcond_cloud(:) = (/ cloud_sm(ld), cloud_sm(rd) /)

  stationid(1:n_tot-1) = stationid(2:n_tot)
  year(1:n_tot-1) = year(2:n_tot)
  month(1:n_tot-1) = month(2:n_tot)
  mprec(1:n_tot-1) = mprec(2:n_tot)
  mwet(1:n_tot-1) = mwet(2:n_tot)
  mtmin(1:n_tot-1) = mtmin(2:n_tot)
  mtmax(1:n_tot-1) = mtmax(2:n_tot)
  mcloud(1:n_tot-1) = mcloud(2:n_tot)
  ndm(1:n_tot-1) = ndm(2:n_tot)
  lat(1:n_tot-1) = lat(2:n_tot)
  lon(1:n_tot-1) = lat(2:n_tot)

  !loop to next station/year/month
  if (end_counter == 0) then
    exit
  end if

end do

close(10)
close(30)

contains

!-------------------------------------

function are_consecutive_months(stationid, year, month) result(r)

character(12), intent(in) :: stationid(:)
integer, intent(in)       :: year(:), month(:)
integer                   :: i, r(size(stationid) - 1)

do i=1,size(r)
  if (stationid(i) /= stationid(i + 1)) then
    r(i) = 0
  else
    if (month(i) == 12) then
      r(i) = l2i(((year(i + 1) == year(i) + 1) .and. (month(i+1) == 1)))
    else
      r(i) = l2i(((year(i + 1) == year(i)) .and. (month(i+1) == month(i) + 1)))
    end if
  end if
end do

end function are_consecutive_months

integer function l2i(l)
  ! convenience function to convert a logical to an integer in [0, 1]

  logical :: l

  if (l) then
    l2i = 1
  else
    l2i = 0
  end if

end function l2i

integer function ndaymonth(year,month)

!return the number of days in a month given the year in calendar years AD and the month number

integer, intent(in) :: year
integer, intent(in) :: month

integer, parameter, dimension(12) :: std_year = [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]
integer, parameter, dimension(12) :: leapyear = [ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]

if (mod(year,400) == 0) then

  ndaymonth = leapyear(month)

else if (mod(year,100) == 0) then

  ndaymonth = std_year(month)

else if (mod(year,4) == 0) then

  ndaymonth = leapyear(month)

else

  ndaymonth = std_year(month)

end if

end function ndaymonth

!-------------------------------------

end program weathergen_precip