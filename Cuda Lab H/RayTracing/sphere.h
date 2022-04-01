#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere : public hitable {
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, vec3 col, vec3 vec, float r) : center(cen), radius(r), colour(col), vector(vec) {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    vec3 center;
    vec3 colour;
    vec3 vector;
    float radius;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.colour = colour;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.colour = colour;
            return true;
        }
    }
    return false;
}


#endif
