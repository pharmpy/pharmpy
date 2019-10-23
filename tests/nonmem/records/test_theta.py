# -*- encoding: utf-8 -*-

def test_constraints(parser):
    recs = parser.parse('$THETA 1')
    rec = recs.records[0]
    c = rec.constraints
    assert 1 == 0


"""
@pytest.mark.usefixtures('create_record')
@pytest.mark.parametrize('buf,thetas', [
    ('THETA 0', np.array((
        Scalar(0),
    ))),
    ('THETA   12.3 \n\n', np.array((
        Scalar(12.3)
    ))),
    ('THETA  (0,0.00469) ; CL', np.array((
        Scalar(0.00469, lower=0),
    ))),
    ('THETA  (0,3) 2 FIXED (0,.6,1) 10 (-INF,-2.7,0) (37 FIXED)', np.array((
        Scalar(3, lower=0), Scalar(2, fix=True), Scalar(0.6, lower=0, upper=1),
        Scalar(10), Scalar(-2.7, upper=0), Scalar(37, fix=True),
    ))),
])
def test_create(create_record, buf, thetas):
    rec = create_record(buf)
    assert rec.name == 'THETA'
    assert_array_equal(rec.thetas, thetas)


def test_create_replicate(create_record):
    single = create_record('THETA 2 2 2 2 (0.001,0.1,1000) (0.001,0.1,1000) (0.001,0.1,1000)'
                           '       (0.5 FIXED) (0.5 FIXED)')
    multi = create_record('THETA (2)x4 (0.001,0.1,1000)x3 (0.5 FIXED)x2')
    assert_array_equal(single.thetas, multi.thetas)


@pytest.mark.usefixtures('create_record')
@pytest.mark.parametrize('buf,n,new_thetas', [
    ('THETA 0', 1, np.array((
        Scalar(1),
    ))),
    ('THETA 0', 1, np.array((
        Scalar(1.23, fix=True, upper=100),
    ))),
    ('THETA 1 2', 2, np.array((
        Scalar(1),
    ))),
    ('THETA 1 2', 2, np.array((
        Scalar(1), Scalar(0, fix=True),
        Scalar(1.2383289E2, lower=9, fix=True),
    ))),
])
def test_replace(create_record, buf, n, new_thetas):
    rec = create_record(buf)
    thetas = rec.thetas
    assert len(thetas) == n

    rec.thetas = new_thetas
    assert_array_equal(rec.thetas, new_thetas)
"""
