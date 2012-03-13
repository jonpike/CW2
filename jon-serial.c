/*
 ** Code to implement a d2q9-bgk lattice boltzmann scheme.
 ** 'd2' inidates a 2-dimensional grid, and
 ** 'q9' indicates 9 velocities per grid cell.
 ** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
 **
 ** The 'speeds' in each cell are numbered as follows:
 **
 ** 6 2 5
 **  \|/
 ** 3-0-1
 **  /|\
 ** 7 4 8
 **
 ** A 2D grid 'unwrapped' in row major order to give a 1D array:
 **
 **           cols
 **       --- --- ---
 **      | D | E | F |
 ** rows  --- --- ---
 **      | A | B | C |
 **       --- --- ---
 **
 **  --- --- --- --- --- ---
 ** | A | B | C | D | E | F |
 **  --- --- --- --- --- ---
 */

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>
#include<omp.h>

#define NSPEEDS         9
#define PARAMFILE       "input.params"
#define OBSTACLEFILE    "obstacles.dat"
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct {
    int    nx;            /* no. of cells in y-deirection */
    int    ny;            /* no. of cells in x-direction */
    int    maxIters;      /* no. of iterations */
    int    reynolds_dim;  /* dimension for Reynolds number */
    double density;       /* density per link */
    double accel;         /* density redistribution */
    double omega;         /* relaxation parameter */
	double accelerate_w1;
	double accelerate_w2;
} t_param;

/* struct to hold the 'speed' values */
typedef struct {
    float speeds[NSPEEDS];
} t_speed;

enum boolean { FALSE, TRUE };

/*
 ** function prototypes
 */

/*
 * Extra Shiz added by Jon
 */

void timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y);

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
void initialise(t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, 
               int** obstacles_ptr, float** av_vels_ptr);

/* 
 ** The main calculation methods.
 ** timestep calls, in order, the functions:
 ** accelerate_flow(), propagate(), rebound() & collision()
 */
void timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
void accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
void propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
void rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
void collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
void write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
void finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
 ** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells);

/* compute average velocity */
double av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char *file);

/*
 ** main program:
 ** initialise, timestep loop, finalise
 */
int main(int argc, char* argv[])
{
    t_param  params;            /* struct to hold parameter values */
    t_speed* cells     = NULL;  /* grid containing fluid densities */
    t_speed* tmp_cells = NULL;  /* scratch space */
    int*     obstacles = NULL;  /* grid indicating which cells are blocked */
    float*  av_vels   = NULL;  /* a record of the av. velocity computed for each timestep */
    int      ii;                /* generic counter */
    
    struct timeval start_time;
    struct timeval end_time;
    struct timeval result_time;
    
    /* initialise our data structures and load values from file */
    initialise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
    
    /* iterate for maxIters timesteps */
    gettimeofday(&start_time, NULL);
    for (ii=0;ii<params.maxIters;ii++) {
        timestep(params,cells,tmp_cells,obstacles);
        av_vels[ii] = av_velocity(params,cells,obstacles);
#ifdef DEBUG
        printf("==timestep: %d==\n",ii);
        printf("av velocity: %.12E\n", av_vels[ii]);
        printf("tot density: %.12E\n",total_density(params,cells));
#endif
    }
    gettimeofday(&end_time, NULL);
    timeval_subtract(&result_time, &end_time, &start_time);
    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t%.12E\n",calc_reynolds(params,cells,obstacles));
    
    printf("Elapsed gettimeofday:\t\t%d s %d us\n", result_time.tv_sec, result_time.tv_usec);
    
    write_values(params,cells,obstacles,av_vels);
    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
    
    return EXIT_SUCCESS;
}

void timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y)
{
    /* Perform the carry for the later subtraction by updating y. */
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (x->tv_usec - y->tv_usec) / 1000000;
        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }
    
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;
}



void timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
    accelerate_flow(params,cells,obstacles);
    propagate(params,cells,tmp_cells);
    rebound(params,cells,tmp_cells,obstacles);
    collision(params,cells,tmp_cells,obstacles);
}

void accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{
    unsigned int ii,jj;     /* generic counters */
    
    /* modify the first column of the grid */
    jj=0;
    //#pragma omp parallel for schedule(static) private(ii) firstprivate(jj) shared(cells)
    for(ii=0;ii<params.ny;ii++) {
        /* if the cell is not occupied and
         ** we don't send a density negative */
		
        if( !obstacles[ii*params.nx + jj] )
		{
			t_speed temp_cell = cells[ii*params.nx + jj];
			
 			if 	((temp_cell.speeds[3] - params.accelerate_w1) > 0.0 &&
           		(temp_cell.speeds[6] - params.accelerate_w2) > 0.0 &&
           		(temp_cell.speeds[7] - params.accelerate_w2) > 0.0 )
			{
            	/* increase 'east-side' densities */
	            temp_cell.speeds[1] += params.accelerate_w1;
	            temp_cell.speeds[5] += params.accelerate_w2;
	            temp_cell.speeds[8] += params.accelerate_w2;
	            /* decrease 'west-side' densities */
	            temp_cell.speeds[3] -= params.accelerate_w1;
	            temp_cell.speeds[6] -= params.accelerate_w2;
	            temp_cell.speeds[7] -= params.accelerate_w2;
	        }
			cells[ii*params.nx + jj] = temp_cell;
		}
    }
}

void propagate(const t_param params, t_speed* cells, t_speed* tmp_cells)
{
    unsigned int ii,jj;            /* generic counters */
    unsigned int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */
    
    /* loop over _all_ cells */
    //#pragma omp parallel for schedule(static) private(ii, jj, x_e, x_w, y_n, y_s) shared(cells, tmp_cells)
    for(ii=0;ii<params.ny;ii++) {
        for(jj=0;jj<params.nx;jj++) {
            /* determine indices of axis-direction neighbours
             ** respecting periodic boundary conditions (wrap around) */
            y_n = (ii + 1) % params.ny;
            x_e = (jj + 1) % params.nx;
            y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
            x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);
            /* propagate densities to neighbouring cells, following
             ** appropriate directions of travel and writing into
             ** scratch space grid */
			t_speed temp_cell = cells[ii*params.nx + jj];
            tmp_cells[ii *params.nx + jj].speeds[0]  = temp_cell.speeds[0]; /* central cell, */
            /* no movement   */
            tmp_cells[ii *params.nx + x_e].speeds[1] = temp_cell.speeds[1]; /* east */
            tmp_cells[y_n*params.nx + jj].speeds[2]  = temp_cell.speeds[2]; /* north */
            tmp_cells[ii *params.nx + x_w].speeds[3] = temp_cell.speeds[3]; /* west */
            tmp_cells[y_s*params.nx + jj].speeds[4]  = temp_cell.speeds[4]; /* south */
            tmp_cells[y_n*params.nx + x_e].speeds[5] = temp_cell.speeds[5]; /* north-east */
            tmp_cells[y_n*params.nx + x_w].speeds[6] = temp_cell.speeds[6]; /* north-west */
            tmp_cells[y_s*params.nx + x_w].speeds[7] = temp_cell.speeds[7]; /* south-west */      
            tmp_cells[y_s*params.nx + x_e].speeds[8] = temp_cell.speeds[8]; /* south-east */      
        }
    }
}

void rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
    unsigned int ii,jj;  /* generic counters */
        /* loop over the cells in the grid */
	//#pragma omp parallel for schedule(static) private(ii, jj) shared(cells, tmp_cells)
        for(ii=0;ii<params.ny;ii++) {
            for(jj=0;jj<params.nx;jj++) {
                /* if the cell contains an obstacle */
                if(obstacles[ii*params.nx + jj]) {
                    /* called after propagate, so taking values from scratch space
                     ** mirroring, and writing into main grid */
					t_speed temp_tmp_cell = tmp_cells[ii*params.nx + jj];
					t_speed temp_cell = cells[ii*params.nx + jj];
                    temp_cell.speeds[1] = temp_tmp_cell.speeds[3];
                    temp_cell.speeds[2] = temp_tmp_cell.speeds[4];
                    temp_cell.speeds[3] = temp_tmp_cell.speeds[1];
                    temp_cell.speeds[4] = temp_tmp_cell.speeds[2];
                    temp_cell.speeds[5] = temp_tmp_cell.speeds[7];
                    temp_cell.speeds[6] = temp_tmp_cell.speeds[8];
                    temp_cell.speeds[7] = temp_tmp_cell.speeds[5];
                    temp_cell.speeds[8] = temp_tmp_cell.speeds[6];
					cells[ii*params.nx + jj] = temp_cell;
                }
            }
        }
}

void collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
    unsigned int ii,jj,kk;                 /* generic counters */
    const float c_sq = 1.0/3.0;  /* square of speed of sound */
    const float w0 = 4.0/9.0;    /* weighting factor */
    const float w1 = 1.0/9.0;    /* weighting factor */
    const float w2 = 1.0/36.0;   /* weighting factor */
    float u_x,u_y;               /* av. velocities in x and y directions */
    float u[NSPEEDS];            /* directional velocities */
    float d_equ[NSPEEDS];        /* equilibrium densities */
    float u_sq;                  /* squared velocity */
    float local_density;         /* sum of densities in a particular cell */
    
    /* loop over the cells in the grid
     ** NB the collision step is called after
     ** the propagate step and so values of interest
     ** are in the scratch-space grid */

    
//#pragma omp parallel for schedule(static) private(ii, jj, kk, u, u_x, u_y, d_equ, u_sq, local_density) shared(cells, tmp_cells)
        for(ii=0	;ii<params.ny;ii++)
        {
            for(jj=0;jj<params.nx;jj++)
            {
                /* don't consider occupied cells */
                if(!obstacles[ii*params.nx + jj])
                {
					t_speed temp_cell = tmp_cells[ii*params.nx + jj];
                    /* compute local density total */
                    local_density = 0.0;
                    for(kk=0;kk<NSPEEDS;kk++)
                    {
                        local_density += temp_cell.speeds[kk];
                    }
                    /* compute x velocity component */
                    u_x = (temp_cell.speeds[1] + 
                           temp_cell.speeds[5] + 
                           temp_cell.speeds[8] -
                           (temp_cell.speeds[3] + 
                            temp_cell.speeds[6] + 
                            temp_cell.speeds[7]))
                    / local_density;
                    /* compute y velocity component */
                    u_y = (temp_cell.speeds[2] + 
                           temp_cell.speeds[5] + 
                           temp_cell.speeds[6]
                           - (temp_cell.speeds[4] + 
                              temp_cell.speeds[7] + 
                              temp_cell.speeds[8]))
                    / local_density;
                    /* velocity squared */ 
                    u_sq = u_x * u_x + u_y * u_y;
                    /* directional velocity components */
                    u[1] =   u_x;        /* east */
                    u[2] =         u_y;  /* north */
                    u[3] = - u_x;        /* west */
                    u[4] =       - u_y;  /* south */
                    u[5] =   u_x + u_y;  /* north-east */
                    u[6] = - u_x + u_y;  /* north-west */
                    u[7] = - u_x - u_y;  /* south-west */
                    u[8] =   u_x - u_y;  /* south-east */
                    /* equilibrium densities */
                    /* zero velocity density: weight w0 */
                    d_equ[0] = w0 * local_density * (1.0 - u_sq / (2.0 * c_sq));
                    /* axis speeds: weight w1 */
                    d_equ[1] = w1 * local_density * (1.0 + u[1] / c_sq
                                                     + (u[1] * u[1]) / (2.0 * c_sq * c_sq)
                                                     - u_sq / (2.0 * c_sq));
                    d_equ[2] = w1 * local_density * (1.0 + u[2] / c_sq
                                                     + (u[2] * u[2]) / (2.0 * c_sq * c_sq)
                                                     - u_sq / (2.0 * c_sq));
                    d_equ[3] = w1 * local_density * (1.0 + u[3] / c_sq
                                                     + (u[3] * u[3]) / (2.0 * c_sq * c_sq)
                                                     - u_sq / (2.0 * c_sq));
                    d_equ[4] = w1 * local_density * (1.0 + u[4] / c_sq
                                                     + (u[4] * u[4]) / (2.0 * c_sq * c_sq)
                                                     - u_sq / (2.0 * c_sq));
                    /* diagonal speeds: weight w2 */
                    d_equ[5] = w2 * local_density * (1.0 + u[5] / c_sq
                                                     + (u[5] * u[5]) / (2.0 * c_sq * c_sq)
                                                     - u_sq / (2.0 * c_sq));
                    d_equ[6] = w2 * local_density * (1.0 + u[6] / c_sq
                                                     + (u[6] * u[6]) / (2.0 * c_sq * c_sq)
                                                     - u_sq / (2.0 * c_sq));
                    d_equ[7] = w2 * local_density * (1.0 + u[7] / c_sq
                                                     + (u[7] * u[7]) / (2.0 * c_sq * c_sq)
                                                     - u_sq / (2.0 * c_sq));
                    d_equ[8] = w2 * local_density * (1.0 + u[8] / c_sq
                                                     + (u[8] * u[8]) / (2.0 * c_sq * c_sq)
                                                     - u_sq / (2.0 * c_sq));
                    /* relaxation step */
					t_speed temp_cell_for_cells = cells[ii*params.nx + jj];
                    for(kk=0;kk<NSPEEDS;kk++)
                    {
                        temp_cell_for_cells.speeds[kk] = (temp_cell.speeds[kk]
                                                               + params.omega * 
                                                               (d_equ[kk] - temp_cell.speeds[kk]));
                    }
					cells[ii*params.nx + jj] = temp_cell_for_cells;
					tmp_cells[ii*params.nx + jj] = temp_cell;
					
                }
            }
        }
}

void initialise(t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, 
               int** obstacles_ptr, float** av_vels_ptr)
{
    FILE   *fp;       /* file pointer */
    int    ii,jj;     /* generic counters */
    int    xx,yy;     /* generic array indices */
    int    blocked;   /* indicates whether a cell is blocked by an obstacle */ 
    int    retval;    /* to hold return value for checking */
    double w0,w1,w2;  /* weighting factors */
    
    /* open the parameter file */
    fp = fopen(PARAMFILE,"r");
    if (fp == NULL) {
        die("could not open file input.params",__LINE__,__FILE__);
    }
    
    /* read in the parameter values */
    retval = fscanf(fp,"%d\n",&(params->nx));
    if(retval != 1) die ("could not read param file: nx",__LINE__,__FILE__);
    retval = fscanf(fp,"%d\n",&(params->ny));
    if(retval != 1) die ("could not read param file: ny",__LINE__,__FILE__);
    retval = fscanf(fp,"%d\n",&(params->maxIters));
    if(retval != 1) die ("could not read param file: maxIters",__LINE__,__FILE__);
    retval = fscanf(fp,"%d\n",&(params->reynolds_dim));
    if(retval != 1) die ("could not read param file: reynolds_dim",__LINE__,__FILE__);
    retval = fscanf(fp,"%lf\n",&(params->density));
    if(retval != 1) die ("could not read param file: density",__LINE__,__FILE__);
    retval = fscanf(fp,"%lf\n",&(params->accel));
    if(retval != 1) die ("could not read param file: accel",__LINE__,__FILE__);
    retval = fscanf(fp,"%lf\n",&(params->omega));
    if(retval != 1) die ("could not read param file: omega",__LINE__,__FILE__);
    
    /* and close up the file */
    fclose(fp);

	// Calculate these here for accelerate function only once:
	params->accelerate_w1 = params->density * params->accel / 9.0;
	params->accelerate_w2 = params->density * params->accel / 36.0;
    
    /* 
     ** Allocate memory.
     **
     ** Remember C is pass-by-value, so we need to
     ** pass pointers into the initialise function.
     **
     ** NB we are allocating a 1D array, so that the
     ** memory will be contiguous.  We still want to
     ** index this memory as if it were a (row major
     ** ordered) 2D array, however.  We will perform
     ** some arithmetic using the row and column
     ** coordinates, inside the square brackets, when
     ** we want to access elements of this array.
     **
     ** Note also that we are using a structure to
     ** hold an array of 'speeds'.  We will allocate
     ** a 1D array of these structs.
     */
    
    /* main grid */
    *cells_ptr = (t_speed*)malloc(sizeof(t_speed)*(params->ny*params->nx));
    if (*cells_ptr == NULL) 
        die("cannot allocate memory for cells",__LINE__,__FILE__);
    
    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed)*(params->ny*params->nx));
    if (*tmp_cells_ptr == NULL) 
        die("cannot allocate memory for tmp_cells",__LINE__,__FILE__);
    
    /* the map of obstacles */
    *obstacles_ptr = malloc(sizeof(int*)*(params->ny*params->nx));
    if (*obstacles_ptr == NULL) 
        die("cannot allocate column memory for obstacles",__LINE__,__FILE__);
    
    /* initialise densities */
    w0 = params->density * 4.0/9.0;
    w1 = params->density      /9.0;
    w2 = params->density      /36.0;
    
    for(ii=0;ii<params->ny;ii++) {
        for(jj=0;jj<params->nx;jj++) {
            /* centre */
            (*cells_ptr)[ii*params->nx + jj].speeds[0] = w0;
            /* axis directions */
            (*cells_ptr)[ii*params->nx + jj].speeds[1] = w1;
            (*cells_ptr)[ii*params->nx + jj].speeds[2] = w1;
            (*cells_ptr)[ii*params->nx + jj].speeds[3] = w1;
            (*cells_ptr)[ii*params->nx + jj].speeds[4] = w1;
            /* diagonals */
            (*cells_ptr)[ii*params->nx + jj].speeds[5] = w2;
            (*cells_ptr)[ii*params->nx + jj].speeds[6] = w2;
            (*cells_ptr)[ii*params->nx + jj].speeds[7] = w2;
            (*cells_ptr)[ii*params->nx + jj].speeds[8] = w2;
        }
    }
    
    /* first set all cells in obstacle array to zero */ 
    for(ii=0;ii<params->ny;ii++) {
        for(jj=0;jj<params->nx;jj++) {
            (*obstacles_ptr)[ii*params->nx + jj] = 0;
        }
    }
    
    /* open the obstacle data file */
    fp = fopen(OBSTACLEFILE,"r");
    if (fp == NULL) {
        die("could not open file obstacles",__LINE__,__FILE__);
    }
    
    /* read-in the blocked cells list */
    while( (retval = fscanf(fp,"%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
        /* some checks */
        if ( retval != 3)
            die("expected 3 values per line in obstacle file",__LINE__,__FILE__);
        if ( xx<0 || xx>params->nx-1 )
            die("obstacle x-coord out of range",__LINE__,__FILE__);
        if ( yy<0 || yy>params->ny-1 )
            die("obstacle y-coord out of range",__LINE__,__FILE__);
        if ( blocked != 1 ) 
            die("obstacle blocked value should be 1",__LINE__,__FILE__);
        /* assign to array */
        (*obstacles_ptr)[yy*params->nx + xx] = blocked;
    }
    
    /* and close the file */
    fclose(fp);
    
    /* 
     ** allocate space to hold a record of the avarage velocities computed 
     ** at each timestep
     */
    *av_vels_ptr = (double*)malloc(sizeof(double)*params->maxIters);
}

void finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
    /* 
     ** free up allocated memory
     */
    free(*cells_ptr);
    *cells_ptr = NULL;
    
    free(*tmp_cells_ptr);
    *tmp_cells_ptr = NULL;
    
    free(*obstacles_ptr);
    *obstacles_ptr = NULL;
    
    free(*av_vels_ptr);
    *av_vels_ptr = NULL;
}

double av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
    int    ii,jj,kk;       /* generic counters */
    int    tot_cells = 0;  /* no. of cells used in calculation */
    double local_density;  /* total density in cell */
    double tot_u_x;        /* accumulated x-components of velocity */
    
    /* initialise */
    tot_u_x = 0.0;
    
    /* loop over all non-blocked cells */
    for(ii=0;ii<params.ny;ii++) {
        for(jj=0;jj<params.nx;jj++) {
            /* ignore occupied cells */
            if(!obstacles[ii*params.nx + jj]) {
                /* local density total */
                local_density = 0.0;
                for(kk=0;kk<NSPEEDS;kk++) {
                    local_density += cells[ii*params.nx + jj].speeds[kk];
                }
                /* x-component of velocity */
                tot_u_x += (cells[ii*params.nx + jj].speeds[1] + 
                            cells[ii*params.nx + jj].speeds[5] + 
                            cells[ii*params.nx + jj].speeds[8]
                            - (cells[ii*params.nx + jj].speeds[3] + 
                               cells[ii*params.nx + jj].speeds[6] + 
                               cells[ii*params.nx + jj].speeds[7])) / 
                local_density;
                /* increase counter of inspected cells */
                ++tot_cells;
            }
        }
    }
    
    return tot_u_x / (double)tot_cells;
}

double calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
    const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
    
    return av_velocity(params,cells,obstacles) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells)
{
    int ii,jj,kk;        /* generic counters */
    double total = 0.0;  /* accumulator */
    
    for(ii=0;ii<params.ny;ii++) {
        for(jj=0;jj<params.nx;jj++) {
            for(kk=0;kk<NSPEEDS;kk++) {
                total += cells[ii*params.nx + jj].speeds[kk];
            }
        }
    }
    
    return total;
}

void write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
    FILE* fp;                     /* file pointer */
    int ii,jj,kk;                 /* generic counters */
    const double c_sq = 1.0/3.0;  /* sq. of speed of sound */
    double local_density;         /* per grid cell sum of densities */
    double pressure;              /* fluid pressure in grid cell */
    double u_x;                   /* x-component of velocity in grid cell */
    double u_y;                   /* y-component of velocity in grid cell */
    
    fp = fopen(FINALSTATEFILE,"w");
    if (fp == NULL) {
        die("could not open file output file",__LINE__,__FILE__);
    }
    
    for(ii=0;ii<params.ny;ii++) {
        for(jj=0;jj<params.nx;jj++) {
            /* an occupied cell */
            if(obstacles[ii*params.nx + jj]) {
                u_x = u_y = 0.0;
                pressure = params.density * c_sq;
            }
            /* no obstacle */
            else {
                local_density = 0.0;
                for(kk=0;kk<NSPEEDS;kk++) {
                    local_density += cells[ii*params.nx + jj].speeds[kk];
                }
                /* compute x velocity component */
                u_x = (cells[ii*params.nx + jj].speeds[1] + 
                       cells[ii*params.nx + jj].speeds[5] +
                       cells[ii*params.nx + jj].speeds[8]
                       - (cells[ii*params.nx + jj].speeds[3] + 
                          cells[ii*params.nx + jj].speeds[6] + 
                          cells[ii*params.nx + jj].speeds[7]))
                / local_density;
                /* compute y velocity component */
                u_y = (cells[ii*params.nx + jj].speeds[2] + 
                       cells[ii*params.nx + jj].speeds[5] + 
                       cells[ii*params.nx + jj].speeds[6]
                       - (cells[ii*params.nx + jj].speeds[4] + 
                          cells[ii*params.nx + jj].speeds[7] + 
                          cells[ii*params.nx + jj].speeds[8]))
                / local_density;
                /* compute pressure */
                pressure = local_density * c_sq;
            }
            /* write to file */
            fprintf(fp,"%d %d %.12E %.12E %.12E %d\n",ii,jj,u_x,u_y,pressure,obstacles[ii*params.nx + jj]);
        }
    }
    
    fclose(fp);
    
    fp = fopen(AVVELSFILE,"w");
    if (fp == NULL) {
        die("could not open file output file",__LINE__,__FILE__);
    }
    for (ii=0;ii<params.maxIters;ii++) {
        fprintf(fp,"%d:\t%.12E\n", ii, av_vels[ii]);
    }
    
    fclose(fp);
}

void die(const char* message, const int line, const char *file)
{
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n",message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}
