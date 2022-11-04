## basic module for physics simulations

import numpy as np
import cv2 as cv
import warnings

np.seterr(all='raise')


STATIC_UID = 1
CONST_VACUUM_PERMI = 8.8541878128*10**(-12) #F/m
CONST_DT_INTERVAL = 10**(-9) #s
CONST_MIN_PARTICLE_SEPARATION = 10**(-3)

npri = np.random.randint



def crossP(x, y):
    return np.cross(x,y)

def dotP(x, y):
    return np.dot(x,y)

def CVector(x=0,y=0,z=0, r=False):
    if r == True:
        return np.array(
            (npri(0, 1e8)-5e7, npri(0, 1e8)-5e7, npri(0, 1e8)-5e7)
            , dtype='float64')
    else:
        return np.array((x,y,z), dtype='float64')

def CVN(vec):
    #normalization func
    vec /= np.sqrt(np.dot(vec, vec))
    return vec

def getUID(classidx='UNKNOWN'):
    idsalt=np.random.randint(1e3, 1e8)
    return "{}_{}".format(idsalt, classidx)


class particle:
    
    def __init__(self, pos=CVector(0), vel=CVector(0), charge=0, puid='', Qp = 0, Qpmfield = [0, CVector(0)] ):
        self.uid = getUID("{}particle".format(puid))
        self.killed = 0
        self.pos = pos
        self.vel = vel
        self.charge = charge
        self.mass = 1

        self.Qp = Qp
        self.Qpmfield = Qpmfield

        self.FF = CVector(0)
        self.dpos = CVector(0)
        self.dvel = CVector(0) 
        
        self.idx_painted = 0
        self.idx_updated = 0
        self.is_painted = False
        self.is_updated = True


        ####
        ####self.PaintObjs = updatePaintObj()
        fact = abs(self.charge)/self.mass
        cmax = min( 200, fact*1.8)

        if self.charge < 0:
            color = [ int(cmax + 50) , int(min(200, 18*self.mass)), 20]
        else:
            color = [ 20, int(min(200, 18*self.mass)), int(cmax + 50) ]
            
        self.PaintObjs = [['point' , [color, 1+1*self.mass] ]]
        #######

    def accel(self, FF, Dt=CONST_DT_INTERVAL):
        FF /= self.mass
        self.FF = FF
        self.dpos = self.vel*Dt + 0.5*FF*(Dt**2)
        self.dvel = FF*Dt
        self.is_updated = False

    #def Qaccel(self, FF, Dt=CONST_DT_INTERVAL):
        #self.FF = 
        
        
    def update(self):
        self.pos += self.dpos
        self.vel += self.dvel
        self.idx_updated += 1
        self.is_updated = True
        self.is_painted = False

    def collision(self, q):
        self.charge += q.charge
        self.vel += q.vel
        self.mass += q.mass
        updatePaintObj()

    def kill(self):
        self.killed = 1

    def getPrintCoor(self):
        return self.pos

    def updatePaintObj(self):
        fact = abs(self.charge)/self.mass
        cmax = min( 200, fact*1.8)

        if self.charge < 0:
            color = [ int(cmax + 50) , int(min(200, 18*self.mass)), 20]
        else:
            color = [ 20, int(min(200, 18*self.mass)), int(cmax + 50) ]
            
        self.PaintObjs = [['point' , [color, 1+1*self.mass] ]]
        
    def getPaintObjs(self):
        return self.PaintObjs

    def desc(self, v=0):
        if self.killed:
            kmsg = 'killed'
        else:
            kmsg = '0'
        if v == 0:
            print((self.uid, kmsg, self.pos, self.vel, self.charge
              , self.FF, self.dpos, self.dvel
              , self.idx_painted, self.idx_updated
              , self.is_painted, self.is_updated))
        else:
            print("\n----------------------------------------")
            print(">> uid: {}-{}\n\n>> pos: {}\n>> vel: {}\n"
                  .format(self.uid, kmsg, self.pos, self.vel))
            print(">> charge: {}\n>> FF: {}\n>> dpos: {}"
                  .format(self.charge, self.FF, self.dpos))
            print(">> dvel: {}\n\n>> idx_painted: {}\n>> idx_updated: {}\n"
                  .format(self.dvel, self.idx_painted, self.idx_updated))
            print(">> is_painted: {}\n>> self.is_updated: {}\n"
                  .format(self.is_painted, self.is_updated))
            print("----------------------------------------\n")
            
            
        
    

def FieldForce(q, setQ, epsilon = CONST_VACUUM_PERMI, envfs = None):
    E=CVector(0,0,0)
    envFF=CVector(0,0,0)

    if envfs:
        try:
           for e in envfs:
                envFF += e(q)
        except:
            print('>> >> Numpy RuntimeError encountered or something BAD')
            print('>> >> KILLING PARTICLE AT FieldForce()[envfs[e:{}]]:'
                  .format(e.__name__), end='')
            q.kill()
            q.desc(v=1)
            setQ.remove(q)
            return 0
    
    Qs = setQ.copy()
    Qs.remove(q)
    
    for Q in Qs:
        try:
            r_Qq = q.pos - Q.pos
            #impliment divide by zero protection
            m_r = np.sqrt(dotP(r_Qq, r_Qq))
            if m_r > CONST_MIN_PARTICLE_SEPARATION:
                E += (r_Qq)*Q.charge/m_r**3
            else:
                q.collision(Q)
                print('>> Collision {} and {}:'.format(q.uid, Q,uid))
                print('>> >> KILLING PARTICLE AT FieldForce()[COLLISION]:', end='')
                Q.kill()
                Q.desc(v=1)
                setQ.remove(Q)
        except:
            print('>> >> Numpy RuntimeError encountered or something BAD')
    
            if sum(abs(Q.pos)) > sum(abs(q.pos)):
                print('>> >> KILLING PARTICLE AT FieldForce()[particle_interact[particle]]:', end='')
                Q.kill()
                Q.desc(v=1)
                setQ.remove(Q)
            else:
                print('>> >> KILLING PARTICLE AT FieldForce()[particle_interact[SELF]]:', end='')
                q.kill()
                q.desc(v=1)
                setQ.remove(q)
                return 0
    return E/(4*np.pi*epsilon) + envFF

class Camera:
    def __init__(self, shape, pos = CVector(-430, -430, 110), orien = CVN(CVector(1,1,0))):
        self.pos = pos
        self.orien = orien
        #rot is fixed for now
        #implement 3d rotation matrix
        self.rot = CVN(CVector(1,-1,0))
        
        self.scr_shape = shape
        self.scr_hdiag = 0.5*np.sqrt(self.scr_shape[0]**2+self.scr_shape[1]**2)
        self.fov_scrscale = 20

        self.focalangle = np.pi/3
        self.focaldis = self.scr_hdiag/self.fov_scrscale/np.tan(np.pi/6)

        self.grid_mat = self.getGrid()
        self.vel = CVector()

    def identMat(i):
        return np.array((i[0],0,0),(0,i[1],0),(0,0,i[2]))
    
    '''

    def getRotMat( ai, aj, ak, vi, vj, vk ):
        
        Ri = [[1,            0,           0],\
              [0,   np.cos(ai), -np.sin(ai)],\
              [0,   np.sin(ai),  np.cos(ai)]]
        
        Rj = [[np.cos(aj),  0, np.sin(aj)],\
              [0,           1,          0],\
              [-np.sin(aj), 0, np.cos(ai)]]
        
        Rk = [[np.cos(ak), -np.sin(ak), 0],\
              [np.sin(ak),  np.cos(ak), 0],\
              [         0,           0, 1]]

        
        ri = np.matmul(Ri, identMat(vi))
        rj = np.matmul(Rj, identMat(vj))
        rk = np.matmul(Rk, identMat(vk))

        return np.matmul(np.matmul(ri, rj), np.rk)

    ''                       
    def _move_mat(self, trans=CVector(), rot=CVector()):
        rotP = np.cross(self.rot, self.orien)  

        self.pos += trans[0]*self.rot + trans[1]*self.orien + trans[2]*rotP

        if not np.sum(rot):
            pass

        sinr = np.sin(rot)
        cosr = np.cos(rot)
        
        
        #yaw == rot[0]
        if rot[0]:
            Rx = [[1,0,0],[0,cosr]]
            _orien = self.orien.copy()
            #self.orien = self.orien + sinr[0]*self.rot - (1-cosr[0])*self.orien
            self.orien = sinr[0]*self.rot + cosr[0]*self.orien
            #self.rot = self.rot + sinr[0]*self.orien - (1-cosr[0])*self.rot
            self.rot = sinr[0]*_orien + cosr[0]*self.rot

            #update
            self.orien = CVN(self.orien)
            self.rot = CVN(self.rot)           

        
        #pitch == rot[1]
        if rot[1]:
            rotP = np.cross(self.rot, self.orien)
            #self.orien = self.orien + sinr[1]*rotP - (1-cosr[1])*self.orien
            self.orien = sinr[1]*rotP + cosr[1]*self.orien

            #update
            self.orien = CVN(self.orien)
        
        
        #roll == rot[2]
        if rot[2]:
            rotP = np.cross(self.rot, self.orien)
            self.rot = sinr[2]*rotP + cosr[2]*self.rot

            #update
            self.rot = CVN(self.rot)

        #making self.rot perpendicular to self.orien
        self.rot -= np.dot(self.rot, self.orien)
        self.rot = CVN(self.rot)
    '''

    def move(self, trans=CVector(), rot=CVector()):
        rotP = CVN(np.cross(self.rot, self.orien))

        #print(trans, rot)
        self.vel *= 0.995
        trans = np.round(self.vel, 4)

        #print(trans, rot)
        self.pos += trans[0]*self.rot + trans[1]*self.orien + trans[2]*rotP

        if not np.sum(rot):
            pass

        sinr = np.sin(rot)
        cosr = np.cos(rot)
        
        
        #yaw == rot[0]
        if rot[0]:
            _orien = self.orien.copy()
            #self.orien = self.orien + sinr[0]*self.rot - (1-cosr[0])*self.orien
            self.orien = sinr[0]*self.rot + cosr[0]*self.orien
            #self.rot = self.rot + sinr[0]*self.orien - (1-cosr[0])*self.rot
            self.rot = sinr[0]*_orien + cosr[0]*self.rot

            #update
            self.orien = CVN(self.orien)
            #making self.rot perpendicular to self.orien
            self.rot -= np.dot(self.rot, self.orien)*self.orien
            self.rot = CVN(self.rot)     

        
        #pitch == rot[1]
        if rot[1]:
            rotP = CVN(np.cross(self.rot, self.orien))
            #self.orien = self.orien + sinr[1]*rotP - (1-cosr[1])*self.orien
            self.orien = sinr[1]*rotP + cosr[1]*self.orien

            #update
            self.orien -= np.dot(self.orien, self.rot)*self.rot
            self.orien = CVN(self.orien)
        
        
        #roll == rot[2]
        if rot[2]:
            rotP = CVN(np.cross(self.rot, self.orien))
            self.rot = sinr[2]*rotP + cosr[2]*self.rot

            #update
            #making self.rot perpendicular to self.orien
            self.rot -= np.dot(self.rot, self.orien)*self.orien
            self.rot = CVN(self.rot)

        
        
    def update(self, d_fdis = 0, d_fangle = 0):
        # tan(focalangle) = (scr_hdiag/fov_scrscale) / focaldis
        if d_fangle:
            self.focalangle += d_fangle
            self.fov_scrscale = self.scr_hdiag/self.focaldis/np.tan(self.focalangle)
        elif d_fdis:
            self.focaldis += d_fdis
            self.fov_scrscale = self.scr_hdiag/self.focaldis/np.tan(self.focalangle)
            #self.focalangle = np.tanh(self.scr_hdiag/self.fov_scrscale/self.focaldis)

    def getCamCo(self, rcoor, chk_frame = True):
        fcoor = (rcoor - (self.pos - self.orien*self.focaldis))

        #in view chk
        orien_proj = dotP(fcoor, self.orien)
        if orien_proj < self.focaldis or (chk_frame and orien_proj/np.sqrt(np.dot(fcoor,fcoor)) < np.cos(self.focalangle)):
            return None

        #coor in view
        lamda = self.focaldis/orien_proj

        scri = lamda*fcoor - self.focaldis*self.orien

        X = np.dot(scri, self.rot) * self.fov_scrscale
        y = np.cross(scri, self.rot)
        if np.dot(y, self.orien) > 0:
            sign = -1
        else:
            sign = 1
        Y = sign * np.sqrt(np.dot(y, y)) * self.fov_scrscale
        
        return ( ( X , Y ), orien_proj )

    
    def getScrCo(self, rcoor):
        ccoor = self.getCamCo(rcoor)

        if ccoor == None:
            return None
        
        c2 = ( int(ccoor[0][0] + self.scr_shape[0]/2), int(ccoor[0][1] + self.scr_shape[1]/2) )
        
        if c2[0]<0 or c2[0]>self.scr_shape[0]-1 or c2[1]<0 or c2[1]>self.scr_shape[1]-1:
            return None
        else:
            return ( (c2[0], c2[1]) , ccoor[1] )
    

    def getScrLi(self, rcoors):
        ccoor = [ self.getCamCo(c, chk_frame=False) for c in rcoors ]

        if None in ccoor:
            return None           
        
        c1 = ( int(ccoor[0][0][0] + self.scr_shape[0]/2), int(ccoor[0][0][1] + self.scr_shape[1]/2) )
        c2 = ( int(ccoor[1][0][0] + self.scr_shape[0]/2), int(ccoor[1][0][1] + self.scr_shape[1]/2) )

        retval, c1, c2 =  cv.clipLine( (0,0, self.scr_shape[0], self.scr_shape[1]),\
                                       c1, c2)
        
        if retval:
            return ( (c1, c2) , int((ccoor[0][1]+ccoor[1][1])/2) )
        else:
            return None

    def getScrPl(self, rcoors):
        #implement
        pass

    def getScrCo_ORTHO_XY(self, pos):
        c2 = pos[:2]
        c2 = (c2 + self.off)*self.mul
        c2 = (c2 + [self.scr_shape[0]/2, self.scr_shape[1]/2])
        
        if c2[0]<0 or c2[0]>self.scr_shape[0]-1 or c2[1]<0 or c2[1]>self.scr_shape[1]-1:
            return False
        else:
            return (int(c2[0]), int(c2[1]))

    def getGrid(self):
        #print("grid")
        pass
        sr, er, d = -100, 100, 10
        pos = self.pos
        pos = ( int(pos[0]), int(pos[1]), int(pos[2]) )
        
        grid_coor = []

        for i in range(pos[0]+sr, pos[0]+er, d):
            for j in range(pos[1]+sr, pos[1]+er, d):
                for k in range(pos[2]+sr, pos[2]+er, d):
                    pass
                    

        return grid_coor

    def printStats(self, scr):
        font = cv.FONT_HERSHEY_PLAIN
        
        cv.putText(scr, 'pos: {}    orien: {}    rot: {}'\
                   .format(np.round_(self.pos),      np.round_(self.orien, 3),       np.round_(self.rot, 3))
                   , (20, scr.shape[1] - 40), font, 1, (250, 250, 250), 1)

        cv.putText(scr, 'focaldis: {}  focalangle: {}  fov_scrscale: {}  vel: {}'\
                   .format(np.round_(self.focaldis),    np.round_(self.focalangle, 4),  np.round_(self.fov_scrscale, 2),\
                           np.round_(self.vel, 2))
                   , (20, scr.shape[1] - 20), font, 1, (250, 250, 250), 1)

        #cartesian axis
        ca = np.array([ [10,0,0], [0,10,0], [0,0,10], [0,0,0] ], dtype='int')
        ca = ca + self.pos + self.orien*50
        #print('>>> rcoor\n', ca)
        ca = [ self.getCamCo(c)[0] for c in ca ]
        #print('>>> ccoor\n', ca)
        ca = np.array(ca, dtype='int')
        ca = ca - ca[3]

        #print('>>> before size\n', ca)
        s = [ np.dot(ca[i], ca[i]) for i in range(3) ]
        s = np.sqrt(s)
        size = max(s)
        #print('>>> sizing\n', s, 'size', size)

        disp_radius = 30
        disp_center = np.array((50, scr.shape[1]-90), dtype='int')

        sa = ca * disp_radius/(size+1) + disp_center

        sa = sa.astype('int')

        #print('>>> sized\n', sa)
        cv.circle(scr, (disp_center[0], disp_center[1]), disp_radius, (150, 150, 150), 0 )
        cv.line(scr, (disp_center[0], disp_center[1]), (sa[0][0], sa[0][1]), (250, 0, 0), 1)
        cv.line(scr, (disp_center[0], disp_center[1]), (sa[1][0], sa[1][1]), (0, 200, 0), 1)
        cv.line(scr, (disp_center[0], disp_center[1]), (sa[2][0], sa[2][1]), (0, 0, 200), 1)
        cv.putText(scr, 'X', (disp_center[0] + disp_radius + 10, disp_center[1]+5), font, 1, (200, 0, 0) )
        cv.putText(scr, 'Y', (disp_center[0] + disp_radius + 20, disp_center[1]+5), font, 1, (0, 200, 0) )
        cv.putText(scr, 'Z', (disp_center[0] + disp_radius + 30, disp_center[1]+5), font, 1, (0, 0, 200) )

        #cv.waitKey(0)
        return scr
            
            
    
class Canvas:
    def __init__(self, shape=(1080, 1080, 3)):
        self.scr_shape=shape
        self.screen = np.zeros(self.scr_shape, dtype='uint8')
        self.camera = Camera(shape)
        
        self.pause_sim = False
        self.time = 0
        self.ppf = 5 # (ns pause per frame)

        self.fade = 0

    def setFade(self, f):
        self.fade = max(0, min(f, 255) )
        
    def getTime(self):
        return self.time

    def paintGrid(self):
        pass
        grid = self.camera.grid_mat
        for l in grid:
            if l[0] and l[1]:
                self.paintLine(l, ((40, 100, 0), 1))
        
    def putStats(self, scr):
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(scr, 'frame: {} sec'.format(self.time)
                   , (20, 50), font, 1, (250, 250, 250), 1)
        self.camera.printStats(scr)
        return scr

    def paintLine(self, coor, extra):
        #implement edge
        size_factor = self.camera.focaldis * self.camera.fov_scrscale/coor[1] 
        self.screen = cv.line(self.screen, coor[0][0], coor[0][1], extra[0], max(1, int(extra[1]*size_factor)) )

    def paintPoint(self, coor, extra):
        size_factor = self.camera.focaldis * self.camera.fov_scrscale/coor[1]
        rad = int( size_factor*extra[1] )
        
        self.screen = cv.circle(self.screen, coor[0], rad, extra[0], -1)
        
    def paint(self, setQ):
        #self.screen = np.zeros(self.scr_shape, dtype='uint8')
        self.screen = (self.screen*(self.fade/255)).astype('uint8')

        for q in setQ:
            qcoor = q.getPrintCoor()
            paint_objs = q.getPaintObjs()
            
            for po in paint_objs:
                if po[0] == 'line':
                    coor = self.camera.getScrLi(qcoor)
                    if coor:
                        self.paintLine( coor, po[1] )

                elif po[0] == 'point':
                    coor = self.camera.getScrCo(qcoor)
                    if coor:
                        self.paintPoint( coor, po[1] )

        #implement grid
        
        dis = self.putStats(self.screen.copy())
        cv.imshow('main', dis)
        
        key = cv.waitKey(self.ppf)
        self.control(key)
        
        return key
    
    
    def __dp_paint(self, setQ):
        self.screen = np.zeros(self.scr_shape, dtype='uint8')
        #self.screen = (self.screen*0.99).astype('uint8')
        rad = 1+int(abs(self.mul[0]/8))

        for q in setQ:
            coor = self.camera.getScrCo_ORTHO_XY(q.pos)
            if coor:            
                if q.charge > 0:
                    color = CONST_COLOR_RED
                elif q.charge == 0:
                    color = CONST_COLOR_GREY
                else:
                    color = CONST_COLOR_BLUE
                    
                self.screen = cv.circle(self.screen, coor, rad, color, -1)

        #implement grid
        
        dis = self.putStats(self.screen.copy())
        cv.imshow('main', dis)
        
        key = cv.waitKey(self.ppf)

        self.control(key)
        return key

    def control(self, key):
        self.camera.move()
        
        if key == -1:
            return

        
        #focaldis
        if key == ord('M'):
            self.camera.update(d_fangle = 0.017)
        elif key == ord('m'):
            self.camera.update(d_fangle = -0.017)
        #focaldis
        if key == ord('N'):
            self.camera.update(d_fdis = 1)
        elif key == ord('n'):
            self.camera.update(d_fdis = -1)

        #translation
        

        elif key == ord('w'):
            self.camera.vel[1]+=0.1
        elif key == ord('s'):
            self.camera.vel[1]-=0.1
        elif key == ord('d'):
            self.camera.vel[0]+=0.1
        elif key == ord('a'):
            self.camera.vel[0]-=0.1
        elif key == ord('x'):
            self.camera.vel[2]+=0.1
        elif key == ord('z'):
            self.camera.vel[2]-=0.1

        #rotation
        elif key == ord('i'):
            self.camera.move(rot = (0, 0.017, 0))
        elif key == ord('k'):
            self.camera.move(rot = (0, -0.017, 0))
        elif key == ord('l'):
            self.camera.move(rot = (0.017, 0, 0))
        elif key == ord('j'):
            self.camera.move(rot = (-0.017, 0, 0))
        elif key == ord('o'):
            self.camera.move(rot = (0, 0, 0.017))
        elif key == ord('u'):
            self.camera.move(rot = (0, 0, -0.017))
        elif key == ord(' '):
            self.camera.vel = CVector()
            self.pause_sim = not self.pause_sim
            key=cv.waitKey(0)

        #lets try velocity
        #self.camera.move(trans = self.camera.vel)






