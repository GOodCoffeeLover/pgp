#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <stdlib.h>
#include <time.h>  
typedef unsigned char uchar;

#define sqr3(x) ((x)*(x)*(x))
#define sqr(x) ((x)*(x))

struct item_t {
  float x;
  float y;
  float z;
  float dx;
  float dy;
  float dz;
  float q;
};

const int N = 150; 
const float radius = 1.0, ball_speed=50, ball_q = 3.0; 
item_t item[N];

int w = 1024, h = 648;

float x = -1.5, y = -1.5, z = 1.0;
float dx = 0.0, dy = 0.0, dz = 0.0;
float yaw = 0.0, pitch = 0.0;
float dyaw = 0.0, dpitch = 0.0;

float speed = 0.1;

const float a2 = 15.0;      
const int np = 100;       // Размер текстуры пола

GLUquadric* quadratic;      // quadric объекты - это геометрические фигуры 2-го порядка, т.е. сфера, цилиндр, диск, конус. 

cudaGraphicsResource *res;    
GLuint textures[2];       // Массив из текстурных номеров
GLuint vbo;           // Номер буфера


__global__ void kernel_update_floor(uchar4 *data, item_t* item, int size, float t) { // Генерация текстуры пола на GPU
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int offsetx = blockDim.x * gridDim.x;
  int offsety = blockDim.y * gridDim.y;
  int i, j, k;

  float x, y, fg, fb;
  for(i = idx; i < np; i += offsetx)
    for(j = idy; j < np; j += offsety) {
      x = (2.0 * i / (np - 1.0) - 1.0) * a2;
      y = (2.0 * j / (np - 1.0) - 1.0) * a2;
      //fb = 1000.0 * (sin(0.1 * x * x + t) + cos(0.1 * y * y + t * 0.6) + sin(0.1 * x * x + 0.1 * y * y + t * 0.3));
      fg=0; 
      for(k=0; k<size; ++k)
        fg += 100.0/ (sqr(x - item[k].x) + sqr(y - item[k].y) + sqr(item[k].z) + 0.001);
      fg = min(max(0.0f, fg), 255.0f);
      //fb = min(max(0.0f, fb), 255.0f);
      //data[j * np + i] = make_uchar4((int)fg, (int)(fg + fb)/2, (int)fb, 255);
      switch(size%3){
        case 0:
          data[j * np + i] = make_uchar4((int)fg, 100, 50, 255);
          break;

        case 1:
          data[j * np + i] = make_uchar4(0, (int)(fg), 50, 255);
          break;

        case 2:
          data[j * np + i] = make_uchar4(0, 100, (int)fg, 255);
          break;
      }
      
    }
}

void display() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // Задаем "объектив камеры"
  gluPerspective(90.0f, (GLfloat)w/(GLfloat)h, 0.1f, 100.0f);


  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Задаем позицию и направление камеры
  gluLookAt(x, y, z,
        x + cos(yaw) * cos(pitch),
        y + sin(yaw) * cos(pitch),
        z + sin(pitch),
        0.0f, 0.0f, 1.0f);

  glBindTexture(GL_TEXTURE_2D, textures[0]);  // Задаем текстуру


  static float angle = 0.0;
  for(int i=0; i<N; ++i){
    glPushMatrix();
      glTranslatef(item[i].x, item[i].y, item[i].z); // Задаем координаты центра сферы
      glRotatef(angle, 0.0, 0.0, 1.0);
      gluSphere(quadratic, radius, 32, 32);
    glPopMatrix();
  }
    angle += 0.15;


  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);  // Делаем активным буфер с номером vbo
  glBindTexture(GL_TEXTURE_2D, textures[1]);  // Делаем активной вторую текстуру
  glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)np, (GLsizei)np, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);  // Деактивируем буфер
  // Последний параметр NULL в glTexImage2D говорит о том что данные для текстуры нужно брать из активного буфера
  
  glBegin(GL_QUADS);      // Рисуем пол
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-a2, -a2, 0.0);

    glTexCoord2f(1.0, 0.0);
    glVertex3f(a2, -a2, 0.0);

    glTexCoord2f(1.0, 1.0);
    glVertex3f(a2, a2, 0.0);

    glTexCoord2f(0.0, 1.0);
    glVertex3f(-a2, a2, 0.0);
  glEnd();

  
  glBindTexture(GL_TEXTURE_2D, 0);      // Деактивируем текстуру

  // Отрисовка каркаса куба       
  glLineWidth(2);               // Толщина линий        
  glColor3f(0.5f, 0.5f, 0.5f);        // Цвет линий
  glBegin(GL_LINES);              // Последующие пары вершин будут задавать линии
    glVertex3f(-a2, -a2, 0.0);
    glVertex3f(-a2, -a2, 2.0 * a2);

    glVertex3f(a2, -a2, 0.0);
    glVertex3f(a2, -a2, 2.0 * a2);

    glVertex3f(a2, a2, 0.0);
    glVertex3f(a2, a2, 2.0 * a2);

    glVertex3f(-a2, a2, 0.0);
    glVertex3f(-a2, a2, 2.0 * a2);
  glEnd();

  glBegin(GL_LINE_LOOP);            // Все последующие точки будут соеденены замкнутой линией
    glVertex3f(-a2, -a2, 0.0);
    glVertex3f(a2, -a2, 0.0);
    glVertex3f(a2, a2, 0.0);
    glVertex3f(-a2, a2, 0.0);
  glEnd();

  glBegin(GL_LINE_LOOP);
    glVertex3f(-a2, -a2, 2.0 * a2);
    glVertex3f(a2, -a2, 2.0 * a2);
    glVertex3f(a2, a2, 2.0 * a2);
    glVertex3f(-a2, a2, 2.0 * a2);
  glEnd();

  glColor3f(1.0f, 1.0f, 1.0f);

  glutSwapBuffers();
}

__device__
void move_to_the_box(item_t& item){
  float eps = radius;
  if(item.x > a2-eps){
    item.x = a2-eps;
    item.dx *=-0.9;
  }
  if(item.x < -a2+ eps){
    item.x = -a2 + eps;
    item.dx *=-0.9;
  }

  if(item.y > a2-eps){
    item.y = a2-eps;
    item.dy *=-0.9;
  }
  if(item.y < -a2+ eps){
    item.y = -a2 + eps;
    item.dy *=-0.9;
  }



  if(item.z > 2*a2-eps){
    item.z = 2*a2-eps;
    item.dz *=-0.9;
  }
  if(item.z < 0+ eps){
    item.z = 0 + eps;
    item.dz *=-0.9;
  }

}
__global__ 
void kernel_update_velocity(item_t* item, int size, int x, int y, int z){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  
  float w = 0.9999, e0 = 1e-3, dt = 0.01, K = 50.0, g= 20.0;

  for(int i=idx; i<size; i+=offset){
    // Замедление
    item[i].dx *= w;
    item[i].dy *= w;
    item[i].dz *= w;

    // Отталкивание от стен
    float wall_q = 1.1;
    item[i].dx += wall_q * item[i].q * K * (item[i].x - a2) / (sqr3(fabs(item[i].x - a2)) + e0) * dt;
    item[i].dx += wall_q * item[i].q * K * (item[i].x + a2) / (sqr3(fabs(item[i].x + a2)) + e0) * dt;

    item[i].dy += wall_q * item[i].q * K * (item[i].y - a2) / (sqr3(fabs(item[i].y - a2)) + e0) * dt;
    item[i].dy += wall_q * item[i].q * K * (item[i].y + a2) / (sqr3(fabs(item[i].y + a2)) + e0) * dt;

    item[i].dz += wall_q * item[i].q * K * (item[i].z - 2 * a2) / (sqr3(fabs(item[i].z - 2 * a2)) + e0) * dt;
    item[i].dz += wall_q * item[i].q * K * (item[i].z + 0.0) / (sqr3(fabs(item[i].z + 0.0)) + e0) * dt;

    // Отталкивание от камеры
    float l = sqrt(sqr(item[i].x - x) + sqr(item[i].y - y) + sqr(item[i].z - z));
    item[i].dx += 3.0 * item[i].q * K * (item[i].x - x) / (l * l * l + e0) * dt;
    item[i].dy += 3.0 * item[i].q * K * (item[i].y - y) / (l * l * l + e0) * dt;
    item[i].dz += 3.0 * item[i].q * K * (item[i].z - z) / (l * l * l + e0) * dt;
    
    for(int j=0; j<N; ++j){
      if(i==j)
        continue;
      l = sqrt(sqr(item[i].x - item[j].x) + sqr(item[i].y - item[j].y) + sqr(item[i].z - item[j].z));
      item[i].dx += item[i].q * item[j].q * K * (item[i].x - item[j].x) / (l * l * l + e0) * dt;
      item[i].dy += item[i].q * item[j].q * K * (item[i].y - item[j].y) / (l * l * l + e0) * dt;
      item[i].dz += item[i].q * item[j].q * K * (item[i].z - item[j].z) / (l * l * l + e0) * dt;
    }
    
   
    item[i].dz += -g * dt;
  }  
  __syncthreads();
  for(int i=idx; i<size; i+=offset){
    item[i].x += item[i].dx * dt;
    item[i].y += item[i].dy * dt;
    item[i].z += item[i].dz * dt;
    move_to_the_box(item[i]);
  }

}

void update() {

  float v = sqrt(dx * dx + dy * dy + dz * dz);
  if (v > speed) {    // Ограничение максимальной скорости
    dx *= speed / v;
    dy *= speed / v;
    dz *= speed / v;
  }
  x += dx; dx *= 0.99;
  y += dy; dy *= 0.99;
  z += dz; dz *= 0.99;
  if (z < 1.0) {      // Пол, ниже которого камера не может переместиться
    z = 1.0;
    dz = 0.0;
  }
  if (fabs(dpitch) + fabs(dyaw) > 0.0001) { // Вращение камеры
    yaw += dyaw;
    pitch += dpitch;
    pitch = min(M_PI / 2.0 - 0.0001, max(-M_PI / 2.0 + 0.0001, pitch));
    dyaw = dpitch = 0.0;
  }
  
  item_t *dev_item;
  cudaMalloc(&dev_item, sizeof(item_t)*N);
  cudaMemcpy(dev_item, item, sizeof(item_t)*N, cudaMemcpyHostToDevice);
  
  kernel_update_velocity<<<8, 32>>>(dev_item, N, x, y, z);

  cudaMemcpy(item, dev_item, sizeof(item_t)*N, cudaMemcpyDeviceToHost);

  static float t = 0.0;
  uchar4* dev_data;
  size_t size;
  cudaGraphicsMapResources(1, &res, 0);   // Делаем буфер доступным для CUDA
  cudaGraphicsResourceGetMappedPointer((void**) &dev_data, &size, res); // Получаем указатель на память буфера
  
  

  kernel_update_floor<<<dim3(32, 32), dim3(32, 8)>>>(dev_data, dev_item, N, t);   
  cudaFree(dev_item);

  cudaGraphicsUnmapResources(1, &res, 0);   // Возращаем буфер OpenGL'ю что бы он мог его использовать
  t += 0.01;

  glutPostRedisplay();  // Перерисовка
}

void keys(unsigned char key, int x, int y) {  // Обработка кнопок
  switch (key) {
    case 'w':                 // "W" Движение вперед
      dx += cos(yaw) * cos(pitch) * speed;
      dy += sin(yaw) * cos(pitch) * speed;
      dz += sin(pitch) * speed;
    break;
    case 's':                 // "S" Назад
      dx += -cos(yaw) * cos(pitch) * speed;
      dy += -sin(yaw) * cos(pitch) * speed;
      dz += -sin(pitch) * speed;
    break;
    case 'a':                 // "A" Влево
      dx += -sin(yaw) * speed;
      dy += cos(yaw) * speed;
      break;
    case 'd':                 // "D" Вправо
      dx += sin(yaw) * speed;
      dy += -cos(yaw) * speed;
    break;

    case 'r':
      for(int i=0; i<N; ++i){
        item[i].dx *=0.3;
        item[i].dy *=0.3;
        item[i].dz *=0.3;
      }
    break;

    case 27: //esc
      cudaGraphicsUnregisterResource(res);
      glDeleteTextures(2, textures);
      glDeleteBuffers(1, &vbo);
      gluDeleteQuadric(quadratic);
      exit(0);
    break;
  }
}

void mouse(int x, int y) {
  static int x_prev = w / 2, y_prev = h / 2;
  float dx = 0.005 * (x - x_prev);
    float dy = 0.005 * (y - y_prev);
  dyaw -= dx;
    dpitch -= dy;
  x_prev = x;
  y_prev = y;

  // Перемещаем указатель мышки в центр, когда он достиг границы
  if ((x < 20) || (y < 20) || (x > w - 20) || (y > h - 20)) {
    glutWarpPointer(w / 2, h / 2);
    x_prev = w / 2;
    y_prev = h / 2;
    }
}

void shoot(){
  static int num=0;
  num = (num+1)%N;
  item[num].x = x + 2*cos(yaw) * cos(pitch);
  item[num].y = y + 2*sin(yaw) * cos(pitch);
  item[num].z = z + 2*sin(pitch);

  item[num].dx = ball_speed*cos(yaw) * cos(pitch);
  item[num].dy = ball_speed*sin(yaw) * cos(pitch);
  item[num].dz = ball_speed*sin(pitch);

}

void mouse_pressed(int button, int state, int x, int y) {
  
  if (state != GLUT_DOWN)
      return;
  
  if (button == GLUT_RIGHT_BUTTON) {
      dx *= 0.777;
      dy *= 0.777;
      dz *= 0.777;
  }

  if (button == GLUT_LEFT_BUTTON) {
      shoot();
  } 
}

void reshape(int w_new, int h_new) {
  w = w_new;
  h = h_new;
  glViewport(0, 0, w, h);                                     // Сброс текущей области вывода
  glMatrixMode(GL_PROJECTION);                                // Выбор матрицы проекций
  glLoadIdentity();                                           // Сброс матрицы проекции
}


__host__
int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(w, h);
  glutCreateWindow("OpenGL");

  glutIdleFunc(update);
  glutDisplayFunc(display);
  glutKeyboardFunc(keys);
  glutPassiveMotionFunc(mouse);
  glutMouseFunc(mouse_pressed);
  glutReshapeFunc(reshape);

  glutSetCursor(GLUT_CURSOR_NONE);  // Скрываем курсор мышки

  int wt, ht;
  FILE *in = fopen("in.data", "rb");
  fread(&wt, sizeof(int), 1, in);
  fread(&ht, sizeof(int), 1, in);
  uchar *data = (uchar *)malloc(sizeof(uchar) * wt * ht * 4);
  fread(data, sizeof(uchar), 4 * wt * ht, in);
  fclose(in);

  glGenTextures(2, textures);
  glBindTexture(GL_TEXTURE_2D, textures[0]);
  glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)wt, (GLsizei)ht, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);
  // если полигон, на который наносим текстуру, меньше текстуры
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //GL_LINEAR);  // Интерполяция
  // если больше
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //GL_LINEAR);    
  

  quadratic = gluNewQuadric();
  gluQuadricTexture(quadratic, GL_TRUE);  

  glBindTexture(GL_TEXTURE_2D, textures[1]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // Интерполяция 
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Интерполяция 

  glEnable(GL_TEXTURE_2D);                             // Разрешить наложение текстуры
  glShadeModel(GL_SMOOTH);                             // Разрешение сглаженного закрашивания
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);                // Черный фон
  glClearDepth(1.0f);                                  // Установка буфера глубины
  glDepthFunc(GL_LEQUAL);                              // Тип теста глубины. 
  glEnable(GL_DEPTH_TEST);                       // Включаем тест глубины
  glEnable(GL_CULL_FACE);                        // Режим при котором, тектуры накладываются только с одной стороны

  glewInit();           
  glGenBuffers(1, &vbo);                // Получаем номер буфера
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);      // Делаем его активным
  glBufferData(GL_PIXEL_UNPACK_BUFFER, np * np * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);  // Задаем размер буфера
  cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard);        // Регистрируем буфер для использования его памяти в CUDA
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);      // Деактивируем буфер

  srand(time(NULL));
  for(int i=0; i<N; ++i){
    int r = rand() % int(a2);
    item[i].x =  (2*r - a2) * (i%2? 1: -1); 
    item[i].y =  (2*r - a2) * (i%2?-1:  1);
    item[i].z =  2*r;           
    
    item[i].dx = item[i].dy = item[i].dz = 0.1;
    item[i].q = ball_q;
  }

  glutMainLoop();
}

