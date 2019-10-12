/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
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
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

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
  
  for (int i = 0; i < num_particles; ++i) {
    
    if (fabs(yaw_rate) < 0.0000001) {

      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);

    }
    else {

      particles[i].x += (velocity/yaw_rate) * ( sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta) );
      particles[i].y += (velocity/yaw_rate) * ( cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t) );
      particles[i].theta += yaw_rate*delta_t;

    }
    
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);

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

  for (std::size_t o = 0; o < observations.size(); ++o) {
    // initialize minimum distance with large value
    double min_dist = 100000;
    for (std::size_t p = 0; p < predicted.size(); ++p) {
      double distance = dist(predicted[p].x, predicted[p].y, observations[o].x, observations[o].y);
      // if (dist <= sensor_range) {
        // check if current distance is smaller than all previous distances
        if (distance < min_dist) {
          min_dist = distance;
          // set id from observation to map landmark id
          observations[o].id = predicted[p].id;          
        }
      // }
    }
  }

}

double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2))) + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
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

  double weight_sum = 0;
  // iterate over all particles
  vector<double> gauss_vec;
  for (int p = 0; p < num_particles; ++p) {

    // compute predicted observations from map landmarks within sensor range
    vector<LandmarkObs> predictions;
    // int landmark_id;
    // double landmark_x, landmark_y;
    for (unsigned int m = 0; m < map_landmarks.landmark_list.size(); ++m) {
      auto landmark = map_landmarks.landmark_list[m];
      // landmark_id = map_landmarks.landmark_list[m].id_i;
      // landmark_x = map_landmarks.landmark_list[m].x_f;
      // landmark_y = map_landmarks.landmark_list[m].y_f;
      
      // dist_x = landmark_x - particles[p].x;
      // dist_y = landmark_y - particles[p].y;
      // dist_particle_landmark = sqrt( dist_x*dist_x + dist_y*dist_y );

      double dist_particle_landmark = dist(landmark.x_f, landmark.y_f, particles[p].x, particles[p].y);
      // include landmark into predicted observation if it is in sensor range
      if ( dist_particle_landmark <= sensor_range ) {
        predictions.push_back({landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    // initialize vector for transformed observations of current particle
    vector<LandmarkObs> transformed_observations;

    // iterate over all observations and transform them into map coordinates
    for (unsigned int o = 0; o < observations.size(); ++o) {
      
      double temp_obs_x = cos(particles[p].theta)*observations[o].x - sin(particles[p].theta)*observations[o].y + particles[p].x;
      double temp_obs_y = sin(particles[p].theta)*observations[o].x + cos(particles[p].theta)*observations[o].y + particles[p].y;
      transformed_observations.push_back({observations[o].id, temp_obs_x, temp_obs_y});
    }

    // associate all observations with predicted observations (landmark positions) for current particle
    dataAssociation(predictions, transformed_observations);

    particles[p].weight = 1.0; // delete

    // iterate over all observations
    for (unsigned int o = 0; o < transformed_observations.size(); ++o) {

      for (unsigned int m = 0; m < predictions.size(); ++m) {
        if (predictions[m].id == transformed_observations[o].id) {
    
          double gauss = multiv_prob(std_landmark[0], std_landmark[1], observations[o].x, observations[o].y, predictions[m].x, predictions[m].y);
          particles[p].weight *= gauss;
          gauss_vec.push_back(gauss);
          weight_sum += particles[p].weight;

          particles[p].associations.push_back(predictions[m].id);
          particles[p].sense_x.push_back(predictions[m].x);
          particles[p].sense_y.push_back(predictions[m].y);
          break;
        }
      }

    }
    for (int p = 0; p < num_particles; ++p)
    {
      particles[p].weight /= weight_sum;
    }
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
  for (int p = 0; p < num_particles; ++p) {
    weights.push_back(particles[p].weight);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> discr_distribution(weights.begin(), weights.end());
  
  // initialize new particles
  vector<Particle> new_particles;
  for (int n=0; n<num_particles; ++n) {
    Particle p = particles[discr_distribution(gen)];
    new_particles.push_back({p.id, p.x, p.y, p.theta, p.weight});
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