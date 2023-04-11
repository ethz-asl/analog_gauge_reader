import numpy as np

# Here code for fit_ellipse, cart_to_pol and get_ellipse is taken from following blog post:
# https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
# As described in the blog post this implements a numerically stable version
# of the ellipse fitting, described here https://autotrace.sourceforge.net/WSCG98.pdf

# -------------Fit Ellipse to Points ---------------


def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x * y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    # pylint: disable=unused-variable
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a * c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den

    num = 2 * (a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g)
    fac = np.sqrt((a - c)**2 + 4 * b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp / ap)**2
    if r > 1:
        r = 1 / r

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi / 2
    else:
        phi = np.arctan((2. * b) / (a - c)) / 2
        if a > c:
            phi += np.pi / 2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi / 2
    phi = phi % np.pi

    return x0, y0, ap, bp, phi


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2 * np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y


# ------------------Project Point To Ellipse-----------------------------


def get_polar_angle(point, ellipse_params):
    """
    Formula taken from
    https://www.petercollingridge.co.uk/tutorials/computational-geometry/finding-angle-around-ellipse/
    Important: This doesn't calculate the actual angle
    but the theta in the parametrization x = a*cos(theta), y=b*sin(theta)
    See link if you want to get the actual angle.
    """
    x0, y0, ap, bp, phi = ellipse_params

    # shift point and ellipse to origin
    x_shift = point[0] - x0
    y_shift = point[1] - y0
    point_shift = np.array([x_shift, y_shift])

    # rotate point and ellipse to align it with x and y axis
    R = np.array([[np.cos(-phi), -np.sin(-phi)], [np.sin(-phi), np.cos(-phi)]])
    point_rotate = R @ point_shift
    x_rotate = point_rotate[0]
    y_rotate = point_rotate[1]

    # find angle
    theta = np.arctan2(ap * y_rotate, bp * x_rotate)

    return theta


def get_point_from_angle(theta, ellipse_params):
    x0, y0, ap, bp, phi = ellipse_params

    x_p = ap * np.cos(theta)
    y_p = bp * np.sin(theta)
    projected_point = np.array([x_p, y_p])

    # rotate back
    R_inv = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    rot_p = R_inv @ projected_point

    # shift back
    x_proj = rot_p[0] + x0
    y_proj = rot_p[1] + y0

    return x_proj, y_proj


def project_point(point, ellipse_params):
    theta = get_polar_angle(point, ellipse_params)
    return get_point_from_angle(theta, ellipse_params)


# --------------------Intersect Line and Ellipse------------------------------


def get_line_ellipse_point(line_coeffs, x, ellipse_params):
    """
    Most times you have two intersection points.
    Take the intersection point that has the smallest distance to either start
    or end point of the needle
    :param line_coeffs:
    :param x: x coordinates of end and start point of line
    :param ellipse_params:
    :return:
    """
    intersection_points = find_line_ellipse_intersection(
        line_coeffs, x, ellipse_params)

    if intersection_points.shape[1] == 2:
        line = np.poly1d(line_coeffs)
        y = line(x)
        start_end_points = np.vstack((x, y)).T

        intersection_points = intersection_points.T

        distances = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                distances[i, j] = np.linalg.norm(start_end_points[i] -
                                                 intersection_points[j])

        min_idx = np.unravel_index(distances.argmin(), distances.shape)[1]
        return intersection_points[min_idx]

    return intersection_points


def find_line_ellipse_intersection(line_coeffs, x, ellipse_params):
    """
    If no point exists return empty array with shape (2,0)
    :param line_coeffs:
    :param x: two points on the line
    :param ellipse_params:
    :return: np array with x and y vertically stacked
    """

    x0, y0 = ellipse_params[0:2]
    phi = ellipse_params[4]

    line = np.poly1d(line_coeffs)
    y = line(x)

    x_shift = x - x0
    y_shift = y - y0
    points_shift = np.vstack((x_shift, y_shift))

    # rotate point and ellipse to align it with x and y axis
    R = np.array([[np.cos(-phi), -np.sin(-phi)], [np.sin(-phi), np.cos(-phi)]])
    point_rotate = R @ points_shift
    x_rotate = point_rotate[0, :]
    y_rotate = point_rotate[1, :]

    line_coeffs_rot = np.polyfit(x_rotate, y_rotate, 1)

    intersection_points_centered = find_intersection_points_centered(
        line_coeffs_rot, ellipse_params)

    R_inv = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    intersection_points = R_inv @ intersection_points_centered

    # shift back
    x = intersection_points[0, :] + x0
    y = intersection_points[1, :] + y0

    x_real = x.real[abs(x.imag) < 1e-5]
    y_real = y.real[abs(x.imag) < 1e-5]

    return np.vstack((x_real, y_real))


def find_intersection_points_centered(line_coeffs, ellipse_params):
    line = np.poly1d(line_coeffs)

    ap, bp = ellipse_params[2:4]

    m = line_coeffs[0]
    c = line_coeffs[1]

    a = np.square(ap) * np.square(m) + np.square(bp)
    b = 2 * np.square(ap) * m * c
    c = np.square(ap) * (np.square(c) - np.square(bp))

    x_intersected = np.roots([a, b, c])
    y_intersected = line(x_intersected)

    return np.vstack((x_intersected, y_intersected))
