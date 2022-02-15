#include <stdlib.h>
#include <stdio.h> 
#include <cmath>
#include "tgaimage.h"

typedef unsigned char uchar;

struct uchar4 {
	uchar x;
	uchar y;
	uchar z;
	uchar w;
};

struct vec3 {
	double x;
	double y;
	double z;
};

double dot(vec3 a, vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 prod(vec3 a, vec3 b) {
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

vec3 norm(vec3 v) {
	double l = sqrt(dot(v, v));
	return {v.x / l, v.y / l, v.z / l};
}

vec3 diff(vec3 a, vec3 b) {
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

vec3 add(vec3 a, vec3 b) {
	return {a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
	return {a.x * v.x + b.x * v.y + c.x * v.z,
				  a.y * v.x + b.y * v.y + c.y * v.z,
				  a.z * v.x + b.z * v.y + c.z * v.z};
}

void print(vec3 v) {
	printf("%e %e %e\n", v.x, v.y, v.z);
}

struct trig {
	vec3 a;
	vec3 b;
	vec3 c;
	uchar4 color;
};

trig trigs[6];

void build_space() {

	vec3 floor[4] = {{-5,-5,0}, {-5,5,0},{5,5,0}, {5,-5,0}};

	trigs[0] = {floor[0], floor[3], floor[1], {0, 0, 255, 0}};
	trigs[1] = {floor[2], floor[3], floor[1], {0, 0, 255, 0}};
	
	trigs[2] = {{-2,-2, 4}, {2, -2, 4}, {0, 2, 4}, {128, 0, 128, 0}};
	trigs[3] = {{-2, -2, 4}, {2, -2, 4}, {0, 0, 7}, {255, 0, 0, 0}};
	trigs[4] = {{-2,-2, 4}, {0, 0, 7}, {0, 2, 4}, {255, 255, 0, 0}};
	trigs[5] = {{0, 0, 7}, {2, -2, 4}, {0, 2, 4}, {0, 255, 0, 0}};



}

uchar4 ray(vec3 pos, vec3 dir) {
	int k, k_min = -1;
	double ts_min;
	for(k = 0; k < 6; k++) {
		vec3 e1 = diff(trigs[k].b, trigs[k].a);
		vec3 e2 = diff(trigs[k].c, trigs[k].a);
		vec3 p = prod(dir, e2);
		double div = dot(p, e1);
		if (fabs(div) < 1e-10)
			continue;
		vec3 t = diff(pos, trigs[k].a);
		double u = dot(p, t) / div;
		if (u < 0.0 || u > 1.0)
			continue;
		vec3 q = prod(t, e1);
		double v = dot(q, dir) / div;
		if (v < 0.0 || v + u > 1.0)
			continue;
		double ts = dot(q, e2) / div; 	
		if (ts < 0.0)
			continue;
		if (k_min == -1 || ts < ts_min) {
			k_min = k;
			ts_min = ts;
		}
	}
	if (k_min == -1)
		return {0, 0, 0, 0};

	return trigs[k_min].color;
}

void render(vec3 pc, vec3 pv, int w, int h, double angle, TGAImage& image) {
	int i, j;
	double dw = 2.0 / (w - 1.0);
	double dh = 2.0 / (h - 1.0);
	double z = 1.0 / tan(angle * M_PI / 360.0);
	vec3 bz = norm(diff(pv, pc));
	vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
	vec3 by = norm(prod(bx, bz));
	uchar4 polyg;
	for(i = 0; i < w; i++)	
		for(j = 0; j < h; j++) {
			vec3 v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
			vec3 dir = mult(bx, by, bz, v);
			polyg = ray(pc, norm(dir));
			//data[j*w+i] = ray(pc, dir);
			TGAColor color(
        std::min(polyg.x * 255.0f, 255.0f),
        std::min(polyg.y * 255.0f, 255.0f),
        std::min(polyg.z * 255.0f, 255.0f)
      );
      image.set(i, j, color);
			// print(pc);
			// print(add(pc, dir));
			// printf("\n\n\n");
		}
	// print(pc);
	// print(pv);
	// printf("\n\n\n");
}

int main() {
	double p=1;
	int n=2;
	int k, w = 640*p, h = 480*p;
	char buff[256];
	// uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	vec3 pc, pv;
	TGAImage image(w, h, TGAImage::Format::RGBA);

	build_space();
	
	for(k = 0; k < n; k++) { 
		pc = (vec3) {6.0 * sin(0.05 * k), 6.0 * cos(0.05 * k), 5.0 + 2.0 * sin(0.1 * k)};
		pv = (vec3) {3.0 * sin(0.05 * k + M_PI), 3.0 * cos(0.05 * k + M_PI), 0.0};
		render(pc, pv, w, h, 90.0, image);

		sprintf(buff, "res/%03d.tga", k);
		image.write_tga_file(buff, true, false);
		printf("%d: %s\n", k, buff);
		
	}
	yereturn 0;
}
