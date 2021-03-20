//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Gyenese Mátyás
// Neptun : VSQUVG
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"


// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

const char* const vertexTextureSource = R"(	
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec2 vUV;	// Attrib array 1

	out vec2 texCoord;		//output

	void main() {
		texCoord = vUV;		//copy texture coordinates
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}

)";

const char* const fragmentTextureSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	in vec2 texCoord;
	out vec4 outColor;		// computed color of the current pixel

	vec4 color(vec2 uv){
		return vec4( 1, 0, 0, 1);
	}

	void main() {
		outColor = vec4( 1, 0, 0, 1);//color(texCoord);
	}
)";

GPUProgram gpuProgram[2]; // vertex and fragment shaders
unsigned int vao[2];	   // virtual world on the GPU

/// ---------------------------------------------------------------------------------------------
//new bases

const int g_Points = 51; //50;
const int g_Links = 61; //61;
const float g_IdealDistance = 0.5f;
const float g_Vectorlength = 0.2f;
const float g_Friction = 0.1;
// higheris lower less (steeper)
const float g_StrengthOfAffection = 6; //6  works just fine with 50,61 graph
//higher is bigger (steeper)
const float g_StrengthOfRepulsion = 10; //55


//working numbers:
/*
	6 , 10

*/

float lorentz(vec3 p, vec3 q) {
	return p.x * q.x + p.y * q.y - p.z * q.z;
}

float hDistance(vec3 p, vec3 q) {
	if (-lorentz(q, p) < 1){
		return acoshf(1);
	}
	return acoshf(-lorentz(q, p));
}

//normal vector from p to q
vec3 hNormalVector(vec3 p, vec3 q) {
	float d = hDistance(p, q);
	if (sinhf(d) == 0.0f) {
		return vec3(0, 0, 0);
	}
	vec3 v = (q - p * coshf(d)) / sinhf(d);//sinhf --> lehet 0 akkor mit kell csinálni?
	return v; 
}

vec3 hMirrorpt(vec3 p, vec3 q, float t) {
	vec3 nv = hNormalVector(p, q);
	float d = t * hDistance(p, q);
	if (sinhf(d) == 0.0f) {
		return p;
	}
	return p * coshf(d) + nv * sinhf(d);
}

//mirroring p on q
vec3 hMirror(vec3 p, vec3 q){
	return hMirrorpt(p, q, 2.0f);
}

//get the point halfway between p and q
vec3 hHalfWay(vec3 p, vec3 q) {
	return hMirrorpt(p, q, 0.5f);
}


struct point {
	float x;
	float y;
	point(float X, float Y) {
		x = X;
		y = Y;
	}
	point() {}

};

struct velocity {
	float cosd;
	vec3 v;

	velocity(float c, vec3 vsin) {
		cosd = c;
		v = vsin;
	}
	velocity() {
		cosd = 1.0f;
		v = vec3(0, 0, 0);
	}
	//elõtte
	velocity operator+(const velocity& vel) const { return velocity(cosd * vel.cosd, v * vel.cosd/*ezzel pluszban szorzom*/ + vel.v); }
};

struct line {
	float m;
	float b;
	float p;
	float q;
	line(float x1, float y1, float x2, float y2) {
		if (x1 != x2) {
			if (x1 > x2) {
				m = (y1 - y2) / (x1 - x2);
				p = x1;
				q = x2;
			}
			else {
				m = (y2 - y1) / (x2 - x1);
				p = x2;
				q = x1;
			}
		}
		else {
			//kikötés ha x1 - x2 = 0
			m = 999999999.0f;
			p = x1;
			q = x2;
		}
		b = y1 - (m * x1);
	}

};

bool intersect(line s, line t){
	if (t.m == s.m) {
		return false;
	}
	float x = (s.b - t.b) / (t.m - s.m);
	if (s.p -0.0000001f > x && x > s.q + 0.0000001f && t.p - 0.0000001f > x && x > t.q + 0.0000001f) {
		return true;
	}
	else {
		return false;
	}

}


float affection(float d) {

	
	d= powf(2.0f, d - g_StrengthOfAffection - g_IdealDistance) - powf(2, -g_StrengthOfAffection);
	/*	if (d < 0.004) {
			return 5;
		}*/
	return d;

}

float repulsion(float d) {
	/*if (d == 0) {
		return 0;
	}

	return 1.0f / (g_StrengthOfRepulsion * d) - 1.0f / (g_StrengthOfRepulsion - g_IdealDistance);*/
	return g_StrengthOfRepulsion * powf(d - g_IdealDistance, 2.0f);

}
//vecolcity based functions
velocity hVector(vec3 p, vec3 q, bool neighbour) {
	vec3 nv = hNormalVector(p, q);
	float d = hDistance(p, q);

	//if neighbour
	if (neighbour) {
		//printf("\n %f", d);
		d = d - g_IdealDistance ;
		
		//printf("\n %f \n", d);
	//if not
	}else {
		if ((d - g_IdealDistance) > 0) {
			d = 0;
		}
		else {
			d = d - g_IdealDistance;
		}
	}
	velocity v;

	if (sinhf(d) == 0.0f) {

		return velocity(1.0f, vec3(0, 0, 0));
	}
	//ha nem nulla és pozitív akkor vonzás
	if (d > 0.0f) {
		//TESZT--------------------------------------------------------------------------------
		d = affection(d);
		v = velocity(coshf(d), nv * sinhf(d));
	}else {
		//vec3 h = hHalfWay(p, q);
		vec3 qstroke = hMirrorpt(q, p, 2.0f);
		//qstroke = hMirrorpt(qstroke, p, 2.0f);
		nv = hNormalVector(p, qstroke);
		d = hDistance(p, qstroke);
		//szabad mert ez a rásze lett negatív az elején
		//d = g_IdealDistance -d;
		//TESZT--------------------------------------------------------------------------------
		d = repulsion(d);
		if (sinhf(d) == 0.0f) {

			return velocity(1.0f, vec3(0, 0, 0));
		}
		v = velocity(coshf(d), nv * sinhf(d));
	}
	
	return v;
}
//velocity p->q 
velocity hMirrorvt(vec3 p, vec3 q, float t) {
	vec3 nv = hNormalVector(p, q);
	float d = t * hDistance(p, q);
	//beírás
	if (d > g_Vectorlength) {
		d = g_Vectorlength;
	}
	if (sinhf(d) == 0.0f) {
		return velocity(1.0f, vec3(0, 0, 0));
	}
	return velocity( coshf(d) , nv * sinhf(d));
}

float strengthOfGravityFormula(float d) {
	//return d;
	//return powf(2.0f, 2.f * (d - 3.0f));
	return 0.0001f * d * d;
	//return -1 / ((d - 4.0f) * (d - 4.0f) * (d - 4.0f) * (d - 4.0f));
}

velocity hGravity(vec3 p) {
	vec3 nv = hNormalVector(p, vec3(0, 0, 1));
	float d = hDistance(p, vec3(0, 0, 1));
	return velocity(coshf(strengthOfGravityFormula(d)), nv * sinhf(strengthOfGravityFormula(d)));


}


point g_from = point(0, 0);
point g_to = point(0, 0);
bool pressed = false;
bool right = false;

class Point {
public:
	vec3 m_Coordinates;
	velocity m_Velocity;
	std::vector<Point*> m_Neighbours;


	//constructor to give the point coordinates on the hiperboloid
	Point(vec3 v) {
		if (v.z > sqrtf(3.f)) {
			v.z = sqrtf(3.f);
		}
		if (v.z < 1.0f) {
			v.z = 1.0f;
		}
		m_Coordinates = v;

	}

	//give the point in descartes coordinates
	Point(float x, float y) {
		float z = sqrtf(x * x + y * y + 1.0f);
		m_Coordinates = vec3(x, y, z);

	}

	//give the point coordinates by the screen
	Point(int X, int Y) {
		float x = (float)(2 * X / windowWidth - 1);
		float y = (float)(1 - 2.0f * Y / windowHeight);
		float z = sqrtf(x * x + y * y + 1.0f);
		m_Coordinates = vec3(x, y, z);


	}

	//give a link do we need this?
	void addNeighbour(Point* p) {
		m_Neighbours.push_back(p);
	}

	vec3 homogeneousCoordinates() {
		return m_Coordinates / m_Coordinates.z;
	}

	void pointsInVector(std::vector<float>* v) {
		vec3 p = homogeneousCoordinates();
		v->push_back(p.x);
		v->push_back(p.y);
	}


	void setC(vec3 v) {
		//itt is átírtam
		float z = sqrtf(v.x * v.x + v.y * v.y + 1.0f);
		if (z == 1.0f) {
			v.x = 0;
			v.y = 0;
		}
		m_Coordinates = vec3(v.x, v.y, z);
	}

	bool isNeighbour(Point* p) {
		for (Point* point : m_Neighbours) {
			if (point == p) {
				return true;
			}
		}
		return false;
	}


	void forceOfOthers(Point* point) {
		bool neighbour = isNeighbour(point);//false;
		/*for (Point* p : m_Neighbours) {
			if (p == point) {
				neighbour = true;
			}
		}*/
		setVelocity(hVector(m_Coordinates, point->m_Coordinates, neighbour));
		
	}

	void forceOfOrigo(){
		setVelocity(hGravity(m_Coordinates));
	}

	void hPrint() {
	//	printf("x : %f\ty : %f\tz : %f\n", m_Coordinates.x, m_Coordinates.y, m_Coordinates.z);
	}

	void hVprint() {
		//printf("\n c : %f , v : %f\t %f\t %f\t \n", m_Velocity.cosd, m_Velocity.v.x, m_Velocity.v.y, m_Velocity.v.z);
	}

	void homogeneousPrint() {
		vec3 p = homogeneousCoordinates();
	//	printf("x : %f\ty : %f\tz : %f\n", p.x, p.y, p.z);
	}

	void move() {
		//---------------------------------------------------------------------------
		{
			vec3 p = m_Coordinates * m_Velocity.cosd + m_Velocity.v;
			Point* q = new Point(p.x, p.y);
			//súrlódás -------------------------------------------------------------------------------------------beleírtam pluszba a sebességet csúzli effect ++
			m_Velocity =/* m_Velocity +*/ hMirrorvt(m_Coordinates, q->m_Coordinates, 1.0f - g_Friction);
			delete q; //bele kéne integrálni a plussz velocitit a cuccba
		}
		//---------------------------------------------------------------------------
		vec3 p = hHalfWay(m_Coordinates, m_Coordinates * m_Velocity.cosd + m_Velocity.v);
		Point* h = new Point(p.x, p.y);//------------------------------------------------
		setC(hMirrorpt(m_Coordinates, h->m_Coordinates, 1.0f));
		delete h;
	}

	//----------------------------------------------------------------------------------------ez baj lehet?
	void setVelocity(velocity v) {
		m_Velocity = m_Velocity + v;
		vec3 p = m_Coordinates * m_Velocity.cosd + m_Velocity.v;
		Point* q = new Point(p.x, p.y);
		m_Velocity = hMirrorvt(m_Coordinates, q->m_Coordinates, 1.0f);
		delete q;
		//printf("\n c : %f , v : %f\t %f\t %f\t \n", m_Velocity.cosd, m_Velocity.v.x, m_Velocity.v.y, m_Velocity.v.z);
	}

	vec2 centerOfMass(Point* p) {
		vec3 r = homogeneousCoordinates();
		if (isNeighbour(p)) {
			return vec2(r.x, r.y);
		}
		else {
		return vec2(-r.x, -r.y);
		}

	}

};

class Graph2 {
public:
	std::vector<Point*> m_Points;
	std::vector<int> m_Links;

	void init(int numberOfPoints, int numberOfLinks) {
		srand(2);
		for (int i = 0; i < numberOfPoints; i++) {
			float x = ((float)(rand() % 1000) - 500.0f) / 500.0f;
			float y = ((float)(rand() % 1000) - 500.0f) / 500.0f;

			m_Points.push_back(new Point(x, y));
		}
		srand(3);
		for (int j = 0; j < numberOfLinks; j++) {
			bool hasItAlready = false;
			int x = rand() % numberOfPoints;
			int y = rand() % numberOfPoints;
			if (x != y) {
				for (int i = 0; i < m_Links.size() / 2; i++) {
					if (m_Links[i * 2] == x) {
						if (m_Links[i * 2 + 1] == y) {
							hasItAlready = true;
						}
					}
					if (m_Links[i * 2] == y) {
						if (m_Links[i * 2 + 1] == x) {
							hasItAlready = true;
						}
					}
				}
				if (hasItAlready) {
					j--;
				}
				else {
					m_Links.push_back(x);
					m_Links.push_back(y);
					m_Points[x]->addNeighbour(m_Points[y]);
					m_Points[y]->addNeighbour(m_Points[x]);
				}
			}
			else {
				j--;
			}
		}
		heuristicPlacement();
	}

	void heuristicPlacement(){
		if (g_Points > 5) {
			//sort(); Nem segít
			for (int i = 0; i < 5; i++) {
				float x = ((float)(rand() % 1000) - 500.0f) / 1000.0f;
				float y = ((float)(rand() % 1000) - 500.0f) / 1000.0f;
				delete m_Points[i];
				m_Points[i]= new Point(x, y);
			}

			for (int i = 5; i < g_Points; i++) {
				vec2 sum = vec2(0.0f, 0.0f);
				for (int j = i; j >= 0; j--) {
					sum = sum + m_Points[j]->centerOfMass(m_Points[i]);
				}
				//azért szabad mert újraszámolom a zt szóval az sosem lesz 1 remálhetõleg
				m_Points[i]->setC(vec3(sum.x, sum.y, 1.0f));
			}
		
		}

	}

	int countIntersects() {
		int intersects = 0;
		std::vector<line> lines;
		vec3 p;
		vec3 q;
		for (int i = 0; i < m_Links.size() / 2; i++) {
			p = m_Points[m_Links[2 * i]]->homogeneousCoordinates();
			q = m_Points[m_Links[2 * i + 1]]->homogeneousCoordinates();
			lines.push_back(line(q.x, q.y, p.x, p.y));
		}
		for (int i = 0; i < lines.size(); i++) {
			for (int j = i; j >= 0; j--) {
				if (lines[i].m != lines[j].m) {
					if (intersect(lines[i], lines[j])) {
						intersects++;
					}
				}

			}
		}
		return intersects;
	}

	void sort() {
		for (int i = 0; i < g_Points; i++) {
			int maxn = i;
			for (int j = i; j < g_Points; j++) {
				if (m_Points[j]->m_Neighbours.size() > m_Points[maxn]->m_Neighbours.size()) {
					maxn = j;
				}
			}
			Point* temp = m_Points[i];
			m_Points[i] = m_Points[maxn];
			m_Points[maxn] = temp;
		}
	
	}

	std::vector<float>* getCoordinates(std::vector<float>* v) {
		
		for (int i = 0; i < m_Points.size(); i++) {
			m_Points[i]->pointsInVector(v);
			//printf("[%d]\t", i);
		//	m_Points[i]->hPrint();
		}
		return v;
	}

	std::vector<float>* getLinks(std::vector<float>* v) {
		for (int i : m_Links) {
			m_Points[i]->pointsInVector(v);
		}
		return v;
	}

	void hShift(Point* p, Point* q) {
		vec3 h = hHalfWay(p->m_Coordinates, q->m_Coordinates);
		for (Point* point : m_Points) {
			point->setC(hMirror(point->m_Coordinates, p->m_Coordinates));
			point->setC(hMirror(point->m_Coordinates, h));
		}

	}

	void gShift() {
		Point* p = new Point(vec3(0.0f, 0.0f, 1.0f));
		Point* q = new Point((g_to.x - g_from.x), g_to.y - g_from.y);
		hShift(p,q);
		delete p;
		delete q;
		g_from = g_to;
	}

	void sumVelocity() {
		for (int i = 0; i < m_Points.size(); i++){
			for (int j = 0; j < m_Points.size(); j++) {
				if (i != j) {
					m_Points[i]->forceOfOthers(m_Points[j]);
				}
			}
			m_Points[i]->forceOfOrigo();
			m_Points[i]->hVprint();
		}

		for (auto p : m_Points) {
			p->move();
		}
		
	}



	// OpenGL stuff ---------------------------------------------------------------
	void drawTexture(float r, int smoothness) {
		gpuProgram[1].Use();

		glGenVertexArrays(1, &vao[1]);	// get 1 vao id
		glBindVertexArray(vao[1]);		// make it active

		unsigned int vbo[2];		// vertex buffer object
		glGenBuffers(2, &vbo[0]);	// Generate 1 buffer
		
		float j = 0.0f;
		float k = 0.0f;

		for (Point* p : m_Points) {
			std::vector<float> v;
			std::vector<float> vUV;
			
			for (int i = 0; i < smoothness; i++) {
				float fi = (i * 2 * M_PI / smoothness) + M_PI / 4.0f;
				float x = p->m_Coordinates.x + r * cosf(fi);
				float y = p->m_Coordinates.y + r * sinf(fi);
				float z = sqrtf(x * x + y * y + 1.0f);
				v.push_back(x / z);
				v.push_back(y / z);
				vUV.push_back(j + cosf(fi));
				vUV.push_back(k + sinf(fi));
			}
			j += 1.0f;
			if (j == 8.0f) {
				j = 0.0f;
				k += 1.0f;
			}
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
						  0, 1, 0, 0,    // row-major!
						  0, 0, 1, 0,
						  0, 0, 0, 1 };

		int location = glGetUniformLocation(gpuProgram[0].getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

		glBufferData(GL_ARRAY_BUFFER, 
			v.size() * sizeof(float), &v[0], GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);  
		glVertexAttribPointer(0,  2, GL_FLOAT, GL_FALSE, 0, NULL);
		

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, vUV.size() * sizeof(float), &vUV[0], GL_STATIC_DRAW);	
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, smoothness /*# Elements*/);
		}
	}

	void drawLinks() {
		gpuProgram[0].Use();

		glGenVertexArrays(1, &vao[0]);	// get 1 vao id
		glBindVertexArray(vao[0]);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL);

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		int location = glGetUniformLocation(gpuProgram[0].getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

		std::vector<float> v;

		location = glGetUniformLocation(gpuProgram[0].getId(), "color");
		glUniform3f(location, 1.0f, 1.0f, 0.0f); // 3 floats
		getLinks(&v);

		/*for (auto f : v) {
			printf("%f \n", f);
		}*/
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			v.size() * sizeof(float),  // # bytes
			&v[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later

		glPointSize(10.0f);
		glDrawArrays(GL_LINES, 0 /*startIdx*/, g_Links*2 /*# Elements*/);
	}

	void drawPoints() {
		int location = glGetUniformLocation(gpuProgram[0].getId(), "color");
		glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

		std::vector<float> v;
		getCoordinates(&v);  
		

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			v.size() * sizeof(float),  // # bytes
			&v[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later

		
		glPointSize(10.0f);
		glBindVertexArray(vao[0]);  // Draw call
		glDrawArrays(GL_POINTS, 0 /*startIdx*/, g_Points /*# Elements*/);
	}

	void gDraw() {
		drawLinks();
		//drawPoints();
		drawTexture(0.04f, 4);
	}

};

class UnitCircle {

};

/// ---------------------------------------------------------------------------------------------


Graph2 graph;

// Initialization, create an OpenGL context
void onInitialization() {
	
	glViewport(0, 0, windowWidth, windowHeight);


	graph.init(g_Points, g_Links);

	/*vec3 p = vec3(0.0f, 0.0f, 1.0f);
	vec3 p1 = vec3(0.1f, 0.01f, 1.00995f);
	vec3 p2 = vec3(-5.0f, 5.0f, 7.141428f);

	printf("d1 : %f , d2 : %f", hDistance(p, p1), hDistance(p, p2));*/
	/*
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
	float vertices[] = { -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f };
	
	std::vector<float> v;
	graph.getCoordinates(&v);
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		v.size() * sizeof(float),  // # bytes
		&v[0],	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed
		*/
	// create program for the GPU
	gpuProgram[0].create(vertexSource, fragmentSource, "outColor");
	gpuProgram[1].create(vertexTextureSource, fragmentTextureSource, "outColor");
	gpuProgram[0].Use();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	/*// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
	glPointSize(10.0f);
	glBindVertexArray(vao);  // Draw call
	glDrawArrays(GL_POINTS, 0 /*startIdx*//*, g_Points /*# Elements*///);	
	graph.gDraw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == 'a') { graph.sumVelocity(); glutPostRedisplay();}
	if (key == 's') { graph.heuristicPlacement(); glutPostRedisplay(); }
	if (key == 'c') { printf("Intersects : %d\n", graph.countIntersects()); }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	g_to = point(cX, cY);
	if (pressed && right) {
		graph.gShift();
		glutPostRedisplay();
	}
	/*printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);*/
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; pressed = true ; break;
	case GLUT_UP:   buttonStat = "released"; pressed = false ; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   /*printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);*/ break;
	case GLUT_MIDDLE_BUTTON: /*printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);*/  break;
	case GLUT_RIGHT_BUTTON:  /*printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY)*/; right = pressed; g_from = point(cX, cY); break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
