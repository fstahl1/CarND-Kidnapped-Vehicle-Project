/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <map>


#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std[0]);
  std::normal_distribution<double> dist_y(0, std[1]);
  std::normal_distribution<double> dist_theta(0, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    
    Particle p;
    p.id = i;
    p.x = x + dist_x(gen);
    p.y = y + dist_y(gen);
    p.theta = theta + dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;

  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);  
  
  for (Particle& particle : particles) {   
    if (std::abs(yaw_rate) < 0.0000001) {

      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    }
    else {

      particle.x += (velocity/yaw_rate) * ( sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta) );
      particle.y += (velocity/yaw_rate) * ( cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t) );
      particle.theta += yaw_rate*delta_t;
    }
    
    particle.y += dist_y(gen);
    particle.x += dist_x(gen);
    particle.theta += dist_theta(gen);
  } 
}


void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (LandmarkObs& observation : observations) {

    // initialize minimum distance with large value
    double min_dist = 100000;
    int temp_id = -1;
    
    for (LandmarkObs& pred : predicted) {
      
      double distance = dist(pred.x, pred.y, observation.x, observation.y);     
      // check if current distance is smaller than all previous distances
      if (distance < min_dist) {       
        min_dist = distance;
        // set observation id to map landmark id
        temp_id = pred.id;   
      }
    }
    observation.id = temp_id;
  }
}

double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
  // calculate exponent
  double exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2))) + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // iterate over all particles
  for (Particle& particle : particles) {

    // variables for associations between particles and observations
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    // compute predicted observations from map landmarks within sensor range
    vector<LandmarkObs> predictions;
    for (auto& landmark : map_landmarks.landmark_list) {
  
      double dist_particle_landmark = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
      // include landmark into predicted observation if it is in sensor range
      if ( dist_particle_landmark <= sensor_range ) {
        predictions.push_back({landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    // vector for transformed observations of current particle
    vector<LandmarkObs> transformed_observations;
    // iterate over all observations and transform them into map coordinates
    for (const LandmarkObs& observation : observations) {
      
      double temp_obs_x = cos(particle.theta)*observation.x - sin(particle.theta)*observation.y + particle.x;
      double temp_obs_y = sin(particle.theta)*observation.x + cos(particle.theta)*observation.y + particle.y;
      
      transformed_observations.push_back({observation.id, temp_obs_x, temp_obs_y});
    }

    // associate all observations with predicted observations (landmark positions) for current particle
    dataAssociation(predictions, transformed_observations);

    particle.weight = 1.0;

    // iterate over all observations
    for (LandmarkObs& transformed_observation : transformed_observations) {

      for (LandmarkObs& prediction : predictions) {
        if (prediction.id == transformed_observation.id) {
        
          double gauss = multiv_prob(std_landmark[0], std_landmark[1], transformed_observation.x, transformed_observation.y, prediction.x, prediction.y);
          particle.weight *= gauss;

          // prepare values for particle associations
          associations.push_back(transformed_observation.id);
          sense_x.push_back(transformed_observation.x);
          sense_y.push_back(transformed_observation.y);
          
          break;
        }
      }
    }

    // set associations between current particle and all corresponding observations
    SetAssociations(particle, associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<double> weights;
  for (Particle particle : particles) {
    weights.push_back(particle.weight);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> discr_distribution(weights.begin(), weights.end());
  
  // initialize new particles
  vector<Particle> new_particles(num_particles);
  for (int n=0; n<num_particles; ++n) {
    Particle p = particles[discr_distribution(gen)];
    new_particles[n] = particles[discr_distribution(gen)];
  }

  particles = new_particles;
}  

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}