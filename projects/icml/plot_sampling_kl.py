import pylab
import numpy
from numpy import array, float32

x_axis = [1, 5, 10, 50, 100, 500, 1000]
# MaxOut
y_axis = [array(0.001117859617806971, dtype=float32),
        array(0.0005484464927576482, dtype=float32),
        array(0.00045386978308670223, dtype=float32),
        array(0.000389950058888644, dtype=float32),
        array(0.0003939084417652339, dtype=float32),
        array(0.000375147646991536, dtype=float32),
        array(0.0003697863721754402, dtype=float32)]

y_err = [0.00019057326875627039,
        6.7279025074094525e-05,
        5.3026818763464685e-05,
        4.4121619872748854e-05,
        4.3765298742800949e-05,
        4.0943105425685642e-05,
        3.987132180482149e-05]

pylab.errorbar(x_axis, y_axis, yerr = y_err, label = 'MaxOut')

## tanh
y_axis = [array(0.0017927918815985322, dtype=float32),
        array(0.0007250784547068179, dtype=float32),
        array(0.0005650371895171702, dtype=float32),
        array(0.00046609665150754154, dtype=float32),
        array(0.0004552135069388896, dtype=float32),
        array(0.0004502382653299719, dtype=float32),
        array(0.00044391799019649625, dtype=float32)]

y_err = [0.00023270062021911144,
        7.7557740360498429e-05,
        5.488436529412865e-05,
        4.1577684786170719e-05,
        4.0933111403137443e-05,
        3.9855372440069918e-05,
        3.904469292610884e-05]

pylab.errorbar(x_axis, y_axis, yerr = y_err, label = 'tanh')

pylab.xlabel('# samples')
pylab.ylabel('KL divergence')
pylab.xscale('log')
pylab.title('Averaging Effect on mnist')
pylab.legend()
pylab.show()
