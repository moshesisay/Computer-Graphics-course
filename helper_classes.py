import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    # TODO:
    v = normalize(vector)
    inner = np.dot(v, axis)
    inner = np.multiply(2 * inner, axis)
    v -= inner
    return v

## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        # TODO
        self.direction = np.array(direction)

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        # TODO
        return Ray(intersection_point, normalize(self.direction))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        #TODO
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        #TODO
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        # TODO
        self.position = np.array(position)
        self.direction = np.array(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        #TODO
        return Ray(self.position, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        #TODO
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        #TODO
        d = self.get_distance_from_light(intersection)
        v = normalize(self.position - intersection)
        aff = self.kc + self.kl * d + self.kq * (d ** 2)
        vd = normalize(self.direction)
        inner = np.inner(v, vd)
        return (self.intensity * inner) / aff

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects, current_obj=None):
        intersections = None
        nearest_object = None
        min_distance = np.inf
        #TODO
        for obj in objects:
            if current_obj and obj is current_obj:
                continue
            d = obj.intersect(self)
            if d and d[0] < min_distance:
                min_distance = d[0]
                nearest_object = d[1]
        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection

class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            return t, self
        else:
            return None

    def getOutwardFacingNormal(self, direction, intersectPoint=None):
        norm = self.normal
        point = self.point
        x,y,z = direction + point
        a,b,c = norm
        d = a*point[0] + b*point[1] + c*point[2]
        if (a*x + b*y + c*z - d ) <= 0:
            return norm
        return -norm

class Rectangle(Object3D):
    """
        A rectangle is defined by a list of vertices as follows:
        a _ _ _ _ _ _ _ _ d
         |               |  
         |               |  
         |_ _ _ _ _ _ _ _|
        b                 c
        This function gets the vertices and creates a rectangle object
    """
    def __init__(self, a, b, c, d):
        """
            ul -> bl -> br -> ur
        """
        self.abcd = [np.asarray(v) for v in [a, b, c, d]]
        self.normal = self.compute_normal()

    def compute_normal(self):
        # TODO
        v1 = self.abcd[1] - self.abcd[0]
        v2 = self.abcd[3] - self.abcd[0]
        n = normalize(np.cross(v1, v2))
        return n

    # Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        #TODO
        plane = Plane(self.normal, self.abcd[0])
        distance = plane.intersect(ray)
        if not distance:
            return None
        point = ray.origin + distance[0] * ray.direction
        for i in range(4):
            p1 = self.abcd[i - 1] - point
            p2 = self.abcd[i] - point
            p1xp2 = np.cross(p1, p2)
            if np.dot(self.normal, p1xp2) <= 0:
                return None
        return distance[0], self
    
    def getOutwardFacingNormal(self, direction, intersectPoint=None):
        norm = self.normal
        point = self.abcd[0]
        x,y,z = direction + point
        a,b,c = norm
        d = a*point[0] + b*point[1] + c*point[2]
        if (a*x + b*y + c*z - d ) <= 0:
            return norm
        return -norm

class Cuboid(Object3D):
    def __init__(self, a, b, c, d, e, f):
        """ 
              g+---------+f
              /|        /|
             / |  E C  / |
           a+--|------+d |
            |Dh+------|B +e
            | /  A    | /
            |/     F  |/
           b+--------+/c
        """
        g = np.array(a) + np.array(f) - np.array(d)
        h = np.array(b) + np.array(e) - np.array(c)
        A = Rectangle(a, b, c, d)
        B = Rectangle(d, c, e, f)
        C = Rectangle(f, e, h, g)
        D = Rectangle(g, h, b, a)
        E = Rectangle(g, a, d, f)
        F = Rectangle(h, b, c, e)
        self.face_list = [A,B,C,D,E,F]

    def apply_materials_to_faces(self):
        for t in self.face_list:
            t.set_material(self.ambient,self.diffuse,self.specular,self.shininess,self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both
    def intersect(self, ray: Ray):
        #TODO
        distance = np.inf
        nearest_obj = None
        for rec in self.face_list:
            t = rec.intersect(ray)
            if t and t[0] < distance:
                distance = t[0]
                nearest_obj = t[1]
        return distance, nearest_obj
        

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        #TODO
        a = np.linalg.norm(ray.direction) ** 2
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta <= 0:
            return None
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        t2 = (-b - np.sqrt(delta)) / (2 * a)
        if t1 > 0 and t2 > 0:
            return min(t1, t2), self
        elif t1 > 0 or t2 > 0:
            return max(t1, t2), self
        return None
    
    def getOutwardFacingNormal(self, direction, intersectPoint):
        norm = normalize(intersectPoint - self.center)
        plane = Plane(norm, intersectPoint)
        return plane.getOutwardFacingNormal(direction)
