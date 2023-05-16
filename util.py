import math

def wet_bulb_tw(t, rh):
    a = t * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
    b = math.atan(t + rh) - math.atan(rh - 1.676331)
    c = 0.00391838 * rh ** (1.5) * math.atan(0.023101 * rh)
    d = -4.686035
    tw = a + b + c + d
    return tw


def temperature_wet_bulb(t, rh):
    twb = round(
        t * math.atan(0.151977 * (rh + 8.313659) ** (1 / 2))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * rh ** (3 / 2) * math.atan(0.023101 * rh)
        - 4.686035,
        1,
    )
    return twb  
A=5.391260E-01
B=1.047837E-01
C=-7.493556E-04
D=-1.077432E-03
E=6.414631E-03
F=-5.151526E+00
print(A,':', B,':',C,':',D,':',E,':',F) 


#print((wet_bulb_tw(35,60) * 9 / 5) + 32)

td = 35
rh = 10
print('35c @10RH WebBulb Temp : ', ((A * td + B * rh + C * td * td + D * rh * rh + E * td * rh + F) * 9 / 5) + 32)
print('35c @10RH WebBulb Temp : ',(temperature_wet_bulb(35,10)* 9 / 5) + 32)

td = 35
rh = 50
print('35c @50RH WebBulb Temp : ', ((A * td + B * rh + C * td * td + D * rh * rh + E * td * rh + F) * 9 / 5) + 32)
print('35c @50RH WebBulb Temp : ',(temperature_wet_bulb(35,50)* 9 / 5) + 32)

td = 35
rh = 80
print('35c @80RH WebBulb Temp : ', ((A * td + B * rh + C * td * td + D * rh * rh + E * td * rh + F) * 9 / 5) + 32)
print('35c @80RH WebBulb Temp : ',(temperature_wet_bulb(35,80)* 9 / 5) + 32)

td = 35
rh = 100
print('35c @100RH WebBulb Temp : ', ((A * td + B * rh + C * td * td + D * rh * rh + E * td * rh + F) * 9 / 5) + 32)
print('35c @100RH WebBulb Temp: ',(temperature_wet_bulb(35,100)* 9 / 5) + 32)
