//Standard C libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//Custom library for allocating the 2D array grid
#include "arrays.h"

//C Library of time for random seed usage
#include <time.h>

//Constants of the simulation
const int N_x = 200; //Number of sites in the X direction
const int N_y = 200; //Number of sites in the Y direction

const double max_abs_p = 1.0;
const double min_abs_p = 0.0;

const double max_theta = 10.0;
const double min_theta = -10.0;

const double kappa = 1.0;

const double D_r_par = 1.0;
const double D_r_perp = 1.0;

const double D_theta = 1.0;

const double D_p = 1.0;

const int t_max = 200; //Number of iterations

const double dt = 1.0; //Time step dt, must be an integer as we are considering a discrete system
const double dx = 1.0; //Spacing between each point of the grid in the X direction, must also be an integer as we are considering a discrete system
const double dy = 1.0; //Spacing between each point of the grid in the Y direction, must also be an integer as we are considering a discrete system

//The amount of displacement (read velocity in a classical system), which is the ratio of displecement and time interval must also be an integer

void initial_conditions(double ****rho);
void simulation(double ****rho, double ****rho_new, double **rho_real, FILE *density);

int main (void)
{
  //Creates the arrays used in the simulation
  double ****rho;
  double ****rho_new;
  double **rho_real;

  double n_abs_p, n_theta;
  n_abs_p = (int)(max_abs_p - min_abs_p);
  n_theta = (int)(max_theta - min_theta);
  
  //Creates an output file for the probability densities and simulation storage
  FILE *output;
  create_file(&output, "data.dat", "w");
    
  printf("\n #Created File \n");
  
  //Allocates the previously created arrays using the functions from "arrays.h" library
  printf("\n #Initializing Grid \n");
 
  rho = create_4d_grid(N_x, N_y, n_abs_p, n_theta);
  rho_new = create_4d_grid(N_x, N_y, n_abs_p, n_theta);
  rho_real = create_2d_grid(N_x, N_y);

  printf("\n #Setting Initial Conditions \n");
  initial_conditions(rho);
  
  printf("\n #Running Dynamics \n");
  simulation(rho, rho_new, rho_real, output);
  
  printf("\n #Simulation Finished \n");
  
  return 0;
}

void initial_conditions(double ****rho)
{
  int l, m;

  int n_abs_p, n_theta;
  n_abs_p = (int)(max_abs_p - min_abs_p);
  n_theta = (int)(max_theta - min_theta);

  //Initial conditions of the system
  
  for(l=0 ; l<n_abs_p ; l++){
    for(m=0 ; m<n_theta ; m++){
      /*
      rho[N_x/2+3][N_y/2+3][l][m] = 0.05;
      rho[N_x/2+1][N_y/2+2][l][m] = 0.1;
      rho[N_x/2+2][N_y/2+1][l][m] = 0.1;
      rho[N_x/2+2][N_y/2+2][l][m] = 0.4;
      rho[N_x/2+2][N_y/2+3][l][m] = 0.1;
      rho[N_x/2+3][N_y/2+2][l][m] = 0.1;
      rho[N_x/2+1][N_y/2+1][l][m] = 0.05;
      rho[N_x/2+1][N_y/2+4][l][m] = 0.05;
      rho[N_x/2+4][N_y/2+1][l][m] = 0.05;
      */
      
      //rho[0][3][l][m] = 0.05;
      //rho[1][2][l][m] = 0.1;
      //rho[2][1][l][m] = 0.1;
      //rho[2][2][l][m] = 0.4;
      //rho[2][3][l][m] = 0.1;
      //rho[3][2][l][m] = 0.1;
      //rho[1][1][l][m] = 0.05;
      //rho[1][4][l][m] = 0.05;
      //rho[4][1][l][m] = 0.05;

      rho[N_x/2][N_y/2][l][m] = 1.0;
      //rho[0][0][l][m] = 1.0;
    }
  }
  return ;
}

void simulation(double ****rho, double ****rho_new, double **rho_real, FILE *density)
{
  int t;

  int a, b;

  int dl, dm;
  int i,j,l,m;
  int i_aux, j_aux;
  int i_m_aux, i_p_aux, j_m_aux, j_p_aux; 
  
  int dr_p, dr_m;
  int dp_p, dp_m;
  int i_real, j_real;
  int n_abs_p, n_theta;
  int dtheta_p, dtheta_m;
  int drx_p, drx_m, dry_p, dry_m;

  double R;
  double norm;
  double theta;
  double total;
  double angle;
  double abs_p;
  double v_drift;
  double D_px, D_py;
  
  double zero_sum[N_x][N_y];
  for(i = 0 ; i < N_x ; i++){
    for(j = 0 ; j < N_y ; j++){
      zero_sum[i][j] = 0.0;
    }
  }

  n_abs_p = (int)(max_abs_p - min_abs_p);
  n_theta = (int)(max_theta - min_theta);
   
  //Time loop of the simulation
  for(t = 0 ; t < t_max ; t++ )
    {
      total = 0.0;
      //CONVECTION (DRIFT) in the parallel direction
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  for(l = 0 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){ 
	      v_drift = (int)(kappa * n_abs_p);
	      drx_m = i - v_drift;
	      drx_m = wrap_grid(drx_m,N_x);

	      j_aux = j;
	      
	      if (drx_m*drx_m + j*j > N_x*N_x){
		j_aux = -j;
		drx_m = -drx_m;
	      }

	      j_aux = wrap_grid(j_aux,N_y);
	      drx_m = wrap_grid(drx_m,N_x);
	      
	      rho_new[i][j_aux][l][m] = rho[drx_m][j_aux][l][m];
	    }
	  }
	}
      }
      //Loops for the update of the grid in time T to time T + dt (Sincronous method we update all sites of the grid onto a new grid in a sincronous way)
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  for(l = 0 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){ 
	      rho[i][j][l][m] = rho_new[i][j][l][m];
	    }
	  }
	}
      } 
      
      //DIFFUSION in the perpendicular direction
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  for(l = 0 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){ 
	      dr_m = j-1;
	      dr_m = wrap_grid(dr_m,N_y);
	      
	      dr_p = j+1;
	      dr_p = wrap_grid(dr_p,N_y);

	      i_m_aux = i;
	      i_p_aux = i;
	      
	      if (i*i + dr_m*dr_m > N_x*N_x){
		i_m_aux = -i;
		dr_m = -dr_m;
	      }

	      i_m_aux = wrap_grid(i_m_aux,N_x);
	      dr_m = wrap_grid(dr_m,N_y);

	      if (i*i + dr_p*dr_p > N_x*N_x){
		i_p_aux = -i;
		dr_p = -dr_p;
	      }

	      i_p_aux = wrap_grid(i_p_aux,N_x);
	      dr_p = wrap_grid(dr_p,N_y);
	      
	      rho_new[i][j][l][m] = rho[i][j][l][m] + 0.5*D_r_perp*(rho[i_m_aux][dr_m][l][m] + rho[i_p_aux][dr_p][l][m] - 2.0*rho[i][j][l][m]);
	    }
	  }
	}
      }
      //Loops for the update of the grid in time T to time T + dt (Sincronous method we update all sites of the grid onto a new grid in a sincronous way)
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  for(l = 0 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){ 
	      rho[i][j][l][m] = rho_new[i][j][l][m];
	    }
	  }
	}
      }
      
      //Diffusion in the parallel direction
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  for(l = 0 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){
	      dr_m = i-1;
	      dr_m = wrap_grid(dr_m,N_x);
	      
	      dr_p = i+1;
	      dr_p = wrap_grid(dr_p,N_x);

	      j_m_aux = j;
	      j_p_aux = j;

	      if (dr_m*dr_m + j*j > N_x*N_x){
		j_m_aux = -j;
		dr_m = -dr_m;
	      }

	      j_m_aux = wrap_grid(j_m_aux,N_y);
	      dr_m = wrap_grid(dr_m,N_y);

	      if (dr_p*dr_p + j*j > N_x*N_x){
		j_p_aux = -j;
		dr_p = -dr_p;
	      }

	      j_p_aux = wrap_grid(j_p_aux,N_y);
	      dr_p = wrap_grid(dr_p,N_y);
	      
	      rho_new[i][j][l][m] = rho[i][j][l][m] + 0.5*D_r_par*(rho[dr_m][j_m_aux][l][m] + rho[dr_p][j_p_aux][l][m] - 2.0*rho[i][j][l][m]);
	    }
	  }
	}
      }
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  for(l = 0 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){ 
	      rho[i][j][l][m] = rho_new[i][j][l][m];
	    }
	  }
	}
      }

      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  R = sqrt(1.0*(i-N_x/2.0)*(i-N_x/2.0) + 1.0*(j-N_y/2.0)*(j-N_y/2.0));
	  angle = atan2(1.0*(j-N_y/2.0),1.0*(i-N_x/2.0));

	  for(m = 0 ; m < n_theta ; m++){
	    
	    i_real = (int)(R*cos(-2.0*m*M_PI/n_theta + angle) + 0.5);
	    i_real = wrap_grid(i_real,N_x);
	    
	    j_real = (int)(R*sin(-2.0*m*M_PI/n_theta + angle) + 0.5);
	    j_real = wrap_grid(j_real,N_y);

	    zero_sum[i][j] = zero_sum[i][j] + rho[i][j][0][m];
	  }
	}
      }

      for(i_real = 0 ; i_real < N_x ; i_real++){
	for(j_real = 0 ; j_real < N_y ; j_real++){
	  R = sqrt(1.0*(i_real-N_x/2.0)*(i_real-N_x/2.0) + 1.0*(j_real-N_y/2.0)*(j_real-N_y/2.0));
	  angle = atan2(1.0*(j_real-N_y/2.0),1.0*(i_real-N_x/2.0));

	  for(m = 0 ; m < n_theta ; m++){
	    i = (int)(R*cos(2.0*m*M_PI/n_theta + angle) + 0.5);
	    i = wrap_grid(i,N_x);

	    j = (int)(R*sin(2.0*m*M_PI/n_theta + angle) + 0.5);
	    j = wrap_grid(j,N_y);
	    
	    rho_new[i][j][1][m] = rho[i][j][1][m] + zero_sum[i_real][j_real]/((float)(n_theta));
	  }
	}
      }
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  zero_sum[i][j] = 0.0;
	  for(l = 1 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){ 
	      rho[i][j][l][m] = rho_new[i][j][l][m];
	    }
	  }
	}
      }

      //Diffusion in the polarization absolute val
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  for(l = 1 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){
	      dp_m = l-1;
	      dp_m = wrap_grid(dp_m,n_abs_p);

	      dp_p = l+1;
	      dp_p = wrap_grid(dp_p,n_abs_p);
	      
	      rho_new[i][j][l][m] = rho[i][j][l][m] + 0.5*D_p*(rho[i][j][dp_m][m] + rho[i][j][dp_p][m] - 2.0*rho[i][j][l][m]);
	    }
	  }
	}
      }
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  for(l = 0 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){ 
	      rho[i][j][l][m] = rho_new[i][j][l][m];
	    }
	  }
	}
      }

      //Diffusion in the polarization orientation

      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  
	  R = sqrt(1.0*(i-N_x/2.0)*(i-N_x/2.0) + 1.0*(j-N_y/2.0)*(j-N_y/2.0));
	  angle = atan2(1.0*(j-N_y/2.0),1.0*(i-N_x/2.0));
	  
	  for(l = 1 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){
	    
	      dtheta_m = m-1;
	      dtheta_m = wrap_grid(dtheta_m,n_theta);
	      
	      dtheta_p = m+1;
	      dtheta_p = wrap_grid(dtheta_p,n_theta);

	      if (i==0 && j==0) {
		drx_p = 0;
		drx_m = 0;

		dry_p = 0;
		dry_m = 0;
	      }
	      else {
		drx_p=(int)(R*cos(angle-(dtheta_p-m)*2.0*M_PI/n_theta)+0.5);
		drx_p = wrap_grid(drx_p,N_x);

		dry_p=(int)(R*sin(angle-(dtheta_p-m)*2.0*M_PI/n_theta)+0.5);
		dry_p = wrap_grid(dry_p,N_x);
		
                drx_m=(int)(R*cos(angle-(dtheta_m-m)*2.0*M_PI/n_theta)+0.5);
		drx_m = wrap_grid(drx_m,N_x);
		
                dry_m=(int)(R*sin(angle-(dtheta_m-m)*2.0*M_PI/n_theta)+0.5);
		dry_m = wrap_grid(dry_m,N_x);
	      }

	      if (drx_m*drx_m + dry_m*dry_m > N_x*N_x){
		drx_m = -drx_m;
		dry_m = -dry_m;
	      }

	      drx_m = wrap_grid(drx_m,N_x);
	      dry_m = wrap_grid(dry_m,N_y);

	      if (drx_p*drx_p + dry_p*dry_p > N_x*N_x){
		drx_p = -drx_p;
		dry_p = -dry_p;
	      }

	      drx_p = wrap_grid(drx_p,N_x);
	      dry_p = wrap_grid(dry_p,N_y);
	      
	      rho_new[i][j][l][m] = rho[i][j][l][m] + 0.5*D_theta*(rho[drx_m][dry_m][l][dtheta_m] + rho[drx_p][dry_p][l][dtheta_p] - 2.0*rho[i][j][l][m]);
	    }
	  }
	}
      }
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  for(l = 0 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){ 
	      rho[i][j][l][m] = rho_new[i][j][l][m];
	    }
	  }
	}
      }
      
      //Conversion from the self grid to the real grid
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  R = sqrt(1.0*(i-N_x/2.0)*(i-N_x/2.0) + 1.0*(j-N_y/2.0)*(j-N_y/2.0));
	  angle = atan2(1.0*(j-N_y/2.0),1.0*(i-N_x/2.0));
	  
	  for(l = 0 ; l < n_abs_p ; l++){
	    for(m = 0 ; m < n_theta ; m++){
	      
	      i_real = (int)(R*cos(2.0*m*M_PI/n_theta + angle) + 0.5);
	      i_real = wrap_grid(i_real,N_x);
	      
	      j_real = (int)(R*sin(2.0*m*M_PI/n_theta + angle) + 0.5);
	      j_real = wrap_grid(j_real,N_y);

	      if (i_real*i_real + j_real*j_real > N_x*N_x){
		i_real = -i_real;
		j_real = -j_real;
	      }

	      i_real = wrap_grid(i_real,N_x);
	      j_real = wrap_grid(j_real,N_y);
	      
	      if (i_real <= N_x/2 && j_real <= N_y/2){
		a = (float)i_real + (float)N_x/2;
		b = (float)j_real + (float)N_y/2;
	      }

	      if (i_real <= N_x/2 && j_real > N_y/2 ){
		a = (float)i_real + (float)N_x/2;
		b = (float)j_real - (float)N_x/2;
	      }

	      if (i_real > N_x/2 && j_real <= N_y/2 ){
		a = (float)i_real - (float)N_x/2;
		b = (float)j_real + (float)N_y/2;
	      }
	  
	      if (i_real > N_x/2 && j_real > N_y/2 ){
		a = (float)i_real - (float)N_x/2;
		b = (float)j_real - (float)N_y/2;
	      }

	      a = wrap_grid(a,N_x);
	      b = wrap_grid(b,N_y);

	      rho_real[a][b] = rho_real[a][b] + rho[i][j][l][m];
	    }
	  }
	}
      }
      for(i = 0 ; i < N_x ; i++){
	for(j = 0 ; j < N_y ; j++){
	  fprintf(density,"%d \t %d \t %lf\n", i - N_x/2, j - N_y/2, rho_real[i][j]);

   	  total = total + rho_real[i][j];
	  rho_real[i][j] = 0.0;
	}
      }
      fprintf(density,"\n\n");
      if(t%10 == 0){printf("time=%d \t total_prob=%lf \n",t,total);}
    }
  
  return ;
}
