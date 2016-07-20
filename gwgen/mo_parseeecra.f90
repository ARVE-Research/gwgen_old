module parseeecra

contains

subroutine parse_file( &
    infile, year, nrecords, &
    yr,mn,dy,hr,      & !   year,month,day,hour
    IB,               & !   sky brightness indicator
    LAT,              & !   latitude
    LON,              & !   longitude
    station_id,       & !   land: station number, ship: source deck, ship type
    LO,               & !   land/ocean indicator
    ww,               & !   present weather
    N,                & !   total cloud cover
    Nh,               & !   lower cloud amount
    h,                & !   lower cloud base height
    CL,               & !   low cloud type
    CM,               & !   middle cloud type
    CH,               & !   high cloud type
    AM,               & !   middle cloud amount
    AH,               & !   high cloud amount
    UM,               & !   middle cloud amount
    UH,               & !   high cloud amount
    IC,               & !   change code
    SA,               & !   solar altitude
    RI,               & !   relative lunar illuminance
    SLP,              & !   sea level pressure
    WS,               & !   wind speed
    WD,               & !   wind direction (degrees)
    AT,               & !   air temperature
    DD,               & !   dew point depression
    EL,               & !   Land: station elevation (m)
    IW,               & !   wind speed indicator
    IP)                 !  Land: sea level pressure flag
    !   year,month,day,hour          yr,mn,dy,hr    8   51120100  96113021    none
    !   sky brightness indicator        IB          1          0         1    none
    !   latitude  x100                  LAT         5      -9000      9000    none
    !   longitude x100                  LON         5          0     36000    none
    !   land: station number            ID          5      01000     98999    none
    !   ship: source deck, ship type                        1100      9999  none,9
    !   land/ocean indicator            LO          1          1         2    none
    !   present weather                 ww          2          0        99      -1
    !   total cloud cover               N           1          0         8    none
    !   lower cloud amount              Nh          2         -1         8      -1
    !   lower cloud base height         h           2         -1         9      -1
    !   low cloud type                  CL          2         -1        11      -1
    !   middle cloud type               CM          2         -1        12      -1
    !   high cloud type                 CH          2         -1         9      -1
    !   middle cloud amount x100        AM          3          0       800     900
    !   high cloud amount x100          AH          3          0       800     900
    !   middle cloud amount           UM          1          0         8       9
    !   high cloud amount             UH          1          0         8       9
    !   change code                     IC          2          0         9    none
    !   solar altitude (deg x10)        SA          4       -900       900    none
    !   relative lunar illuminance x100 RI          4       -110       117    none
    !   sea level pressure (mb x10)     SLP         5       9000,L   10999,L    -1
    !   wind speed (ms-1 x10)           WS          3          0       999      -1
    !   wind direction (degrees)        WD          3          0       361      -1
    !   air temperature (C x10)         AT          4       -949,L     599,L   900
    !   dew point depression (C x10)    DD          3          0       700     900
    !   Land: station elevation (m)     EL          4       -350      4877    9000
    !   wind speed indicator            IW          1          0         1       9
    !  Land: sea level pressure flag   IP          1          0)
    implicit none
    
    character(300), intent(in) :: infile
    integer, intent(in) :: nrecords, year
    integer, dimension(nrecords), intent(out) :: &
      yr,mn,dy,hr,      & !   year,month,day,hour
      IB,               & !   sky brightness indicator
      station_id,       & !   land: station number, ship: source deck, ship type
      LO,               & !   land/ocean indicator
      ww,               & !   present weather
      N,                & !   total cloud cover
      Nh,               & !   lower cloud amount
      h,                & !   lower cloud base height
      CL,               & !   low cloud type
      CM,               & !   middle cloud type
      CH,               & !   high cloud type
      UM,               & !   middle cloud amount
      UH,               & !   high cloud amount
      IC,               & !   change code
      WD,               & !   wind direction (degrees)
      EL,               & !   Land: station elevation (m)
      IW,               & !   wind speed indicator
      IP                  !  Land: sea level pressure flag
    real, dimension(nrecords), intent(out) :: &
      LAT,              & !   latitude
      LON,              & !   longitude
      AM,               & !   middle cloud amount
      AH,               & !   high cloud amount
      SA,               & !   solar altitude
      RI,               & !   relative lunar illuminance
      SLP,              & !   sea level pressure
      WS,               & !   wind speed
      AT,               & !   air temperature
      DD                  !   dew point depression
    integer :: i

    open(10,file=infile,status='old')

    20 format(i2,i2,i2,i2, & ! yr,mn,dy,hr
              i1,          & ! IB
              f5.0,          & ! LAT
              f5.0,          & ! LON
              i5,          & ! station_id
              i1,          & ! LO
              i2,          & ! ww
              i1,          & ! N
              i2,          & ! Nh
              i2,          & ! h
              i2,          & ! CL
              i2,          & ! CM
              i2,          & ! CH
              f3.0,          & ! AM
              f3.0,          & ! AH
              i1,          & ! UM
              i1,          & ! UH
              i2,          & ! IC
              f4.0,          & ! SA
              f4.0,          & ! RI
              f5.0,          & ! SLP
              f3.0,          & ! WS
              i3,          & ! WD
              f4.0,          & ! AT
              f3.0,          & ! DD
              i4,          & ! EL
              i1,          & ! IW
              i1)            ! IP
    21 format(i4,i2,i2,i2, & ! yr,mn,dy,hr
              i1,          & ! IB
              f5.0,          & ! LAT
              f5.0,          & ! LON
              i5,          & ! station_id
              i1,          & ! LO
              i2,          & ! ww
              i1,          & ! N
              i2,          & ! Nh
              i2,          & ! h
              i2,          & ! CL
              i2,          & ! CM
              i2,          & ! CH
              f3.0,          & ! AM
              f3.0,          & ! AH
              i1,          & ! UM
              i1,          & ! UH
              i2,          & ! IC
              f4.0,          & ! SA
              f4.0,          & ! RI
              f5.0,          & ! SLP
              f3.0,          & ! WS
              i3,          & ! WD
              f4.0,          & ! AT
              f3.0,          & ! DD
              i4,          & ! EL
              i1,          & ! IW
              i1)            ! IP
    do i=1,nrecords
        if (year < 1997) then
            read(10,20) yr(i),mn(i),dy(i),hr(i),IB(i),LAT(i),LON(i),station_id(i), &
                       LO(i),ww(i),N(i),Nh(i),h(i),CL(i),CM(i),CH(i),AM(i),AH(i), &
                       UM(i),UH(i),IC(i),SA(i),RI(i),SLP(i),WS(i),WD(i),AT(i),DD(i), &
                       EL(i),IW(i),IP(i)
            yr(i) = yr(i) + 1900
        else
            read(10,21) yr(i),mn(i),dy(i),hr(i),IB(i),LAT(i),LON(i),station_id(i), &
                       LO(i),ww(i),N(i),Nh(i),h(i),CL(i),CM(i),CH(i),AM(i),AH(i), &
                       UM(i),UH(i),IC(i),SA(i),RI(i),SLP(i),WS(i),WD(i),AT(i),DD(i), &
                       EL(i),IW(i),IP(i)
        endif
        if (LON(i) > 18000) LON(i) = LON(i) - 36000
        LAT(i) = LAT(i) * 0.01
        LON(i) = LON(i) * 0.01
        AM(i) = AM(i) * 0.01
        AH(i) = AH(i) * 0.01
        SA(i) = SA(i) * 0.01
        RI(i) = RI(i) * 0.01
        SLP(i) = SLP(i) * 0.01
        WS(i) = WS(i) * 0.1
        AT(i) = AT(i) * 0.1
        DD(i) = DD(i) * 0.1
    end do

end subroutine parse_file

end module parseeecra
