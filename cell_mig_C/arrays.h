#include <math.h>

//Function for file creation
int create_file(FILE **file, char file_name[], char * option)
{
  if(strcmp(option,"w")!=0 && strcmp(option,"r")!=0 && strcmp(option,"a")!=0)
    {
      printf("Use a valid option for file operation ((r)ead, (w)rite), (a)ppend");
    }
  *file = fopen(file_name, option);
  return 0;
}

//Function that returns +1 if a value is positive and -1 if not
double sign (double x)
{
  if (x>=0.0)
    {
      return 1.0;
    }
  else
    {
      return -1.0;
    }  
}


//Function that returns a 2D array with N_x * N_y sites
double ** create_2d_grid(int N_x, int N_y)
{ 
  int i, j;
  double **rho; //Creates a 2D double pointer

  //Allocation of a pointer with size N_x times the memory needed for a double type pointer
   rho = (double **) malloc(N_x * sizeof(double *));
   for (i = 0 ; i < N_x+1 ; i++)
     {
       //Allocation of a pointer with size N_y times the memory needed for a double type pointer
       rho[i] = (double *) malloc(N_y * sizeof(double));
     }
   
   
   for (i = 0 ; i < N_x+1 ; i++)
     {
       for (j = 0 ; j < N_y+1 ; j++)
	 {
	   //Zeroes the created 2D pointer (Matrix) 
	     rho[i][j] = 0.0;
	 }
     }

  return rho;
}

//Function that returns a 4D array with N_x * N_y sites and N_x2 * N_y2 auxiliar sites, resulting in N_x * N_y * N_x2 * N_y2 total sites
double **** create_4d_grid(int N_x, int N_y, int N_x2, int N_y2)
{ 
  int i, j, k, l;
  double ****rho; //Creates a 4D array of double type

  //Allocation of a pointer with size N_x times the memory needed for a double type pointer
  rho = (double ****) malloc(N_x * sizeof(double ***));
  for (i = 0 ; i < N_x ; i++)
    {
      //Allocation of a pointer with size N_y times the memory needed for a double type pointer
      rho[i] = (double ***) malloc(N_y * sizeof(double **));

      for (j = 0 ; j < N_y ; j++)
	{
	  rho[i][j] = (double **) malloc(N_x2 * sizeof(double *));
	  
	  for (k = 0 ; k < N_y ; k++)
	    {
	      rho[i][j][k] = (double *) malloc(N_y2 * sizeof(double));
	    }
	}
    }
 
  for (i = 0 ; i < N_x ; i++)
    {
      for (j = 0 ; j < N_y ; j++)
	{
	  for (k = 0 ; k < N_x2 ; k++)
	    {
	      for (l = 0 ; l < N_y2 ; l++)
		{
		  //Zeroes the created 4D pointer (Matrix)
		    rho[i][j][k][l] = 0.0;
		}
	    }
	}
    }
  
  return rho;
}

//Struct that defines a integer point in a cartesian space P(x,y) with P(x,y) = P(P.x,P.y) i.e. P.x = x and P.y = y
struct intStruct
{
  int x;
  int y;
};
//Struct that defines the aforementioned point P(x,y) as a variable datatype
typedef struct intStruct intPoint;

struct doubleStruct
{
  double x;
  double y;
};

typedef struct doubleStruct doublePoint;

//Function that projects a given point on a grid onto another grid given a displacement and rotation between both grids
//The projection is made considering the center of each grid, so that no dissipation between sites occurs
doublePoint changeOfCoordinates (double x0, double y0, double i, double j, double angle)
{
  //x0 and y0 corresponde to the difference in the origins of the plane of reference S and S'
  
  int iNew, jNew;

  intPoint iPoint;
  doublePoint dPoint;

  iNew = round((i-x0)*cos(angle) - (j-y0)*sin(angle));
  jNew = round((i-x0)*sin(angle) + (j-y0)*cos(angle));

  //iNew = round(i*cos(angle) - j*sin(angle)) - x0;
  //jNew = round(i*sin(angle) + j*cos(angle)) - y0;

  //iNew = round(((double)i)*cos(angle) - ((double)j)*sin(angle)) - (double)x0;
  //jNew = round(((double)i)*sin(angle) + ((double)j)*cos(angle)) - (double)y0;

  dPoint.x = iNew;
  dPoint.y = jNew;

  iPoint.x = (int)(iNew);
  iPoint.y = (int)(jNew);
  
  return dPoint;
}


//void sort (int ** array, char choice)
void sort (int * array, int size)
{
  //int size;
  int aux;

  int i, j;
  
  //size = sizeof(array) / sizeof(int);

  // if (choice=="increasing")
  //{
  for (i=0 ; i<size-1 ; i++)
    {
      for (j=0 ; j<size-i-1 ; j++)
	{
	  if (array[j] > array[j+1])
	    {
	      aux = array[j];

	      array[j] = array[j+1];

	      array[j+1] = aux;
	    }
	}
    }
      /*   }
  else
    {
      if (choice=="decreasing")
	{
	  for (i=0 ; i<size-1 ; i++)
	    {
	      for (j=0 ; j<size-i-1 ; j++)
		{
		  if (array[j] > array[j+1])
		    {
		      aux = array[j];

		      array[j] = array[j+1];

		      array[j+1] = aux;
		    }
		}
	    }
	}
      else
	{
	  printf("Error in sorting function from library arrays.h");
	  // break; //Break does not work;
	}
    }
      */
}

double wrap_angle (double theta)
{
  return theta - 2.0*M_PI*floor(theta/(2.0*M_PI));
}

double wrap_grid (double pos, double grid_size)
{
  return pos - grid_size*floor(pos/(grid_size));
}
