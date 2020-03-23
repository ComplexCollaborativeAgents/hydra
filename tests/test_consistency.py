import settings
from os import path
import worlds.science_birds as sb

def read_plan_trace(pathname=
                    path.join(settings.ROOT_PATH, 'data',
                              'science_birds', 'consistency', 'docker_plan_trace.txt')):
    ret = []
    with open(pathname, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith("; TIME:"):
                state = {}
                line = line.lstrip('; ').strip()
                for pair in line.split(', '):
                    value = []
                    try:
                        value = float(pair.split(':')[1])
                    except ValueError:
                        if pair.split(':')[1].strip() == 'false':
                            value = False
                        elif pair.split(':')[1].strip() == 'true':
                            value = True
                        else:
                            assert True
                    except:
                        assert False
                    state[pair.split(':')[0]] = value
                ret.append(state)
            line = f.readline()
    return ret

def test_consistency_checking():
    observations = []
    for i in range(0,7):
        observations.append(sb.SBState.load_from_serialized_state(
            path.join(settings.ROOT_PATH, 'data', 'science_birds', 'serialized_levels', 'dx_test_{}.p'.format(i))))
    assert len(observations) == 7
    plan_trace = read_plan_trace() # list of dictionaries
    assert len(plan_trace) == 20
    print('observations)')
    print(observations[0].summary())
    print('plan_trace')
    print(plan_trace[0])

