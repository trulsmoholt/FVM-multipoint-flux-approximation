from FVMO import run
steps = [1/4,1/8,1/16,1/32,1/64,1/128,1/(2*128),1/(4*128)]
L2_errors = []

for h in steps:
    print('stepsize:    ',h)
    L2_errors.append(run(h))
    if len(L2_errors)>1:
        print('improvement: ',L2_errors[-2]/L2_errors[-1])