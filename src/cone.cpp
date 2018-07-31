/**
* Copyright (C) 2017 Chalmers Revere
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
* USA.
*/

#include "cone.hpp"


Cone::Cone(double x,double y,double z):
  m_pt()
, m_prob()
, m_label()
, m_x(x)
, m_y(y)
, m_z(z)
{
}

double Cone::getX(){
  return m_x;
}

double Cone::getY(){
  return m_y;
}

double Cone::getZ(){

  return m_z;
}

size_t Cone::getLabel(){
  return m_label;
}

void Cone::setX(double x){
  //std::cout << "new x: " << x << " old x: " << m_x << std::endl;
  m_x = x;
}

void Cone::setY(double y){
  //std::cout << "new y: " << y << " old y: " << m_y << std::endl;
  m_y = y;
}
void Cone::setZ(double z){
  m_z = z;
}
void Cone::addHit(){
  m_hits++;
  m_missHit = 0;
}
int Cone::getHits(){
  return m_hits;
}
void Cone::addMiss(){
  m_missHit++;
}

int Cone::getMisses(){
  return m_missHit;
}

bool Cone::isThisMe(double x, double y){
  //double diffX = std::abs(m_x - x);
  //double diffY = std::abs(m_y - y);
  double distance = std::sqrt( (m_x - x)*(m_x - x) + (m_y - y)*(m_y - y) );
  if(distance < 1.5){return true;}else{return false;}
}

bool Cone::checkColor(){
  float totalDetected = (float)(m_orangeCount+m_blueCount+m_yellowCount);
  int currentColorCount = 0;
  if(totalDetected < 2.0f){
    return false;
  }
  if((float)m_blueCount/totalDetected>0.50f){
    m_label=1;
    currentColorCount = m_blueCount;
  }
  else if((float)m_yellowCount/totalDetected>0.50f){
    m_label=2;
    currentColorCount = m_yellowCount;
  }
  else if((float)m_orangeCount/totalDetected>0.0f){
    m_label=4;
    currentColorCount = m_orangeCount;
  }
  if(currentColorCount>1){
    return true;
  }
  else{
    return false;
  }
}

void Cone::addColor(size_t label){
  if(label == 1){
    m_blueCount++;
  }
  else if(label==2){
    m_yellowCount++;
  }
  else if(label == 4){
    m_orangeCount++;
  }
}

bool Cone::shouldBeInFrame(){
  if(m_hits >= 2 && m_y > 0.2 && m_missHit < 2 && m_isValid && checkColor()){
    return true;
  }else{
    return false;
  }
}

bool Cone::shouldBeRemoved(){
  if(m_missHit >= 2 || m_y < 0.2 ){return true;}else{return false;}
}

void Cone::setValidState(bool state){
  m_isValid = state;
}

bool Cone::isValid(){
  return m_isValid;
}