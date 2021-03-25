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

const char * const vertexSource = R"(
	#version 330				
	precision highp float;		

	uniform mat4 MVP;			
	layout(location = 0) in vec2 vp;	

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		
	}
)";

const char * const fragmentSource = R"(
	#version 330			
	precision highp float;	
	
	uniform vec3 color;		
	out vec4 outColor;		

	void main() {
		outColor = vec4(color, 1);	
	}
)";

const char* const vertexTextureSource = R"(	
	#version 330				
	precision highp float;		

	uniform mat4 MVP;			
	layout(location = 0) in vec2 vp;
	layout(location = 1) in vec2 vUV;	

	out vec2 texCoord;		//output

	void main() {
		texCoord = vUV;		//copy texture coordinates
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		
	}

)";

const char* const fragmentTextureSource = R"(
	#version 330			
	precision highp float;	
	
	in vec2 texCoord;
	out vec4 outColor;		

	vec4 color(vec2 uv){
		float x = 1.0f;
		float y = 0.0f;
		float z = 0.0f;
		x = uv.x ;
		y = abs(sin(4.0f * uv.y)*sin(4.0f * uv.y)*sin(200.0f * uv.y)) ;
		z = 1- uv.y;
		return vec4( x, y, z, 1);
	}

	void main() {
		outColor = color(texCoord);
	}
)";

GPUProgram gpuProgram[2]; 
unsigned int vao[2];
unsigned int vbo[2];

const int g_Points = 50; 
const int g_Links = 61; 
const float g_IdealDistance = 0.3f;
const float g_Vectorlength = 0.04f;
const float g_ConstFriction = 0.1;
float g_Friction = 0.1;
const float g_StrengthOfAffection = 3; 
const float g_StrengthOfRepulsion = 10;
int g_SlowEnd = 150;
const int g_End = 150;
bool g_Moved = true;
bool g_RestartNeeded = false;


float lorentz(vec3 p, vec3 q) {
	return p.x * q.x + p.y * q.y - p.z * q.z;
}

float hDistance(vec3 p, vec3 q) {
	if (-lorentz(q, p) < 1){
		return acoshf(1);
	}
	return acoshf(-lorentz(q, p));
}

vec3 hNormalVector(vec3 p, vec3 q) {
	float d = hDistance(p, q);
	if (sinhf(d) == 0.0f) {
		return vec3(0, 0, 0);
	}
	vec3 v = (q - p * coshf(d)) / sinhf(d);
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

vec3 hMirror(vec3 p, vec3 q){
	return hMirrorpt(p, q, 2.0f);
}

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
	velocity operator+(const velocity& vel) const { return velocity(cosd * vel.cosd, v * vel.cosd + vel.v); }
};

float affection(float d) {
	d= powf(2.0f, d - g_StrengthOfAffection - g_IdealDistance) - powf(2, -g_StrengthOfAffection);
	return d;

}

float repulsion(float d) {
	return g_StrengthOfRepulsion * powf(d - g_IdealDistance, 2.0f);

}

velocity hVector(vec3 p, vec3 q, bool neighbour) {
	vec3 nv = hNormalVector(p, q);
	float d = hDistance(p, q);
	bool n = false;
	if (neighbour) {
		d = d - g_IdealDistance ;
	}else {
		if ((d - g_IdealDistance) > 0) {
			d = - 1.0f;
			n = true;
		}
		else {
			d = d - g_IdealDistance;
		}
	}
	velocity v;
	if (sinhf(d) == 0.0f) {
		return velocity(1.0f, vec3(0, 0, 0));
	}
	if (d > 0.0f) {
		d = affection(d);
		v = velocity(coshf(d), nv * sinhf(d));
	}else {
		vec3 qstroke = hMirrorpt(q, p, 2.0f);
		nv = hNormalVector(p, qstroke);
		d = hDistance(p, qstroke);
		if (neighbour) {
			d = repulsion(d);
		}
		else {
			if (d == 0) {
				d = 0.0f;
			}
			else {
				d = 1 / (800.0f * d);
			}
		}
		if (sinhf(d) == 0.0f) {
			return velocity(1.0f, vec3(0, 0, 0));
		}
		v = velocity(coshf(d), nv * sinhf(d));
	}
	return v;
}

velocity hMirrorvt(vec3 p, vec3 q, float t) {
	vec3 nv = hNormalVector(p, q);
	float d = t * hDistance(p, q);
	if (d > g_Vectorlength) {
		d = g_Vectorlength;
	}
	if (sinhf(d) == 0.0f) {
		return velocity(1.0f, vec3(0, 0, 0));
	}
	return velocity( coshf(d) , nv * sinhf(d));
}

float strengthOfGravityFormula(float d) {
	return 0.01f * d * d;
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

	Point(vec3 v) {
		if (v.z > sqrtf(3.f)) {
			v.z = sqrtf(3.f);
		}
		if (v.z < 1.0f) {
			v.z = 1.0f;
		}
		m_Coordinates = v;
	}

	Point(float x, float y) {
		float z = sqrtf(x * x + y * y + 1.0f);
		m_Coordinates = vec3(x, y, z);

	}

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
		bool neighbour = isNeighbour(point);
		setVelocity(hVector(m_Coordinates, point->m_Coordinates, neighbour));
		
	}

	void forceOfOrigo(){
		setVelocity(hGravity(m_Coordinates));
	}

	void move() {
		{
			vec3 p = m_Coordinates * m_Velocity.cosd + m_Velocity.v;
			Point* q = new Point(p.x, p.y);
			m_Velocity = hMirrorvt(m_Coordinates, q->m_Coordinates, 1.0f - g_Friction);
			delete q;
		}	
		vec3 p = hHalfWay(m_Coordinates, m_Coordinates * m_Velocity.cosd + m_Velocity.v);
		Point* h = new Point(p.x, p.y);
		setC(hMirrorpt(m_Coordinates, h->m_Coordinates, 1.0f));
		delete h;
	}

	void setVelocity(velocity v) {
		m_Velocity = m_Velocity + v;
		vec3 p = m_Coordinates * m_Velocity.cosd + m_Velocity.v;
		Point* q = new Point(p.x, p.y);
		m_Velocity = hMirrorvt(m_Coordinates, q->m_Coordinates, 1.0f);
		delete q;
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
		for (int i = 0; i < numberOfPoints; i++) {
			float x = ((float)(rand() % 1000) - 500.0f) / 500.0f;
			float y = ((float)(rand() % 1000) - 500.0f) / 500.0f;

			m_Points.push_back(new Point(x, y));
		}
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
	}

 	void heuristicPlacement(){
		if (g_Points > 5) {
			for (int i = 0; i < 5; i++) {
				float x = (float)(rand() % 1000)  / 1000.0f;
				float y = (float)(rand() % 1000) / 1000.0f;
				delete m_Points[i];
				m_Points[i]= new Point(x, y);
			}

			for (int i = 5; i < g_Points; i++) {
				vec2 sum = vec2(0.0f, 0.0f);
				for (int j = i-1; j >= 0; j--) {
					sum = sum + m_Points[j]->centerOfMass(m_Points[i]);
				}
				m_Points[i]->setC(vec3(sum.x/ (float)i, sum.y/ (float)i));
			}		
		}
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
		}

		for (auto p : m_Points) {
			p->move();
		}	
	}

	~Graph2() {
		for (auto p : m_Points) {
			delete p;
		}	
	}

	void drawTexture(float r, int smoothness) {
		gpuProgram[1].Use();
		glBindVertexArray(vao[1]);

		float j = 0.02f;
		float k = 0.02f;

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
				vUV.push_back(j + cosf(fi)*0.003);
				vUV.push_back(k + sinf(fi) * 0.007);
			}
			j += 0.02f;
			k += 0.15f;
			if (k > 1.0f) {
				k = 0.2f;
			}
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

		float MVPtransf[4][4] = { 1, 0, 0, 0,    
						  0, 1, 0, 0,    
						  0, 0, 1, 0,
						  0, 0, 0, 1 };

		int location = glGetUniformLocation(gpuProgram[0].getId(), "MVP");	
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	

		glBufferData(GL_ARRAY_BUFFER, 
			v.size() * sizeof(float), &v[0], GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);  
		glVertexAttribPointer(0,  2, GL_FLOAT, GL_FALSE, 0, NULL);
		
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, vUV.size() * sizeof(float), &vUV[0], GL_STATIC_DRAW);	
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glDrawArrays(GL_TRIANGLE_FAN, 0 , smoothness);
		}
	}

	void drawLinks() {
		gpuProgram[0].Use();

		glBindVertexArray(vao[0]);		
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

		glEnableVertexAttribArray(0);  
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		float MVPtransf[4][4] = { 1, 0, 0, 0,    
								  0, 1, 0, 0,    
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		int location = glGetUniformLocation(gpuProgram[0].getId(), "MVP");
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	

		std::vector<float> v;

		location = glGetUniformLocation(gpuProgram[0].getId(), "color");
		glUniform3f(location, 1.0f, 1.0f, 0.0f); 

		getLinks(&v);
		glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), &v[0], GL_DYNAMIC_DRAW);	
		glDrawArrays(GL_LINES, 0, g_Links*2);
	}

	void drawPoints() {
		gpuProgram[0].Use();
		glBindVertexArray(vao[0]);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		int location = glGetUniformLocation(gpuProgram[0].getId(), "color");
		glUniform3f(location, 0.70f, 0.70f, 0.70f);
		
		for (Point* p : m_Points) {
			std::vector<float> v;
			
			for (int i = 0; i < 20; i++) {
				float fi = (i * 2 * M_PI / 20) + M_PI / 4.0f;
				float x = p->m_Coordinates.x + 0.045 * cosf(fi);
				float y = p->m_Coordinates.y + 0.045 * sinf(fi);
				float z = sqrtf(x * x + y * y + 1.0f);
				v.push_back(x / z);
				v.push_back(y / z);
			}
		glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), &v[0],GL_DYNAMIC_DRAW);	
		glBindVertexArray(vao[0]); 
		glDrawArrays(GL_TRIANGLE_FAN, 0 , 20 );
		}
	}

	void gDraw() {
		drawLinks();
		drawPoints();
		drawTexture(0.04f, 5);
	}
};

Graph2 graph;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(2, &vao[0]);
	glGenBuffers(2, &vbo[0]);

	graph.init(g_Points, g_Links);

	gpuProgram[0].create(vertexSource, fragmentSource, "outColor");
	gpuProgram[1].create(vertexTextureSource, fragmentTextureSource, "outColor");
	gpuProgram[0].Use();
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);    
	glClear(GL_COLOR_BUFFER_BIT); 
	graph.gDraw();
	glutSwapBuffers(); 
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') { 
		g_Moved = false;
		g_RestartNeeded = true;
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {	
	float cX = 2.0f * pX / windowWidth - 1;	
	float cY = 1.0f - 2.0f * pY / windowHeight;
	g_to = point(cX, cY);
	if (pressed && right) {
		graph.gShift();
		glutPostRedisplay();
	}
}

void onMouse(int button, int state, int pX, int pY) { 	
	float cX = 2.0f * pX / windowWidth - 1;	
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; pressed = true ; break;
	case GLUT_UP:   buttonStat = "released"; pressed = false ; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:    break;
	case GLUT_MIDDLE_BUTTON:   break;
	case GLUT_RIGHT_BUTTON:  ; right = pressed; g_Moved = true; g_from = point(cX, cY); break;
	}
}

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); 
	if (!g_Moved) {
		if (g_RestartNeeded) {
			graph.heuristicPlacement();
			g_RestartNeeded = false;
			g_SlowEnd = g_End;
			g_Friction = g_ConstFriction;
		}
		g_SlowEnd--;
		if (g_SlowEnd < 0) {
			if (g_Friction < 1.0f) {
				g_Friction = g_Friction + 0.005;
			}
		}
		for (int i = 0; i < 5; i++) {
			graph.sumVelocity();
		}
	}
	else {
		g_RestartNeeded = true;
	}
	glutPostRedisplay();
}
