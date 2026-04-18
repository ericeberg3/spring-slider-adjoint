% script to solve spring-slider ODE system

% parameters are stored in data structure M

% rate-and-state parameters
M.f0 = 0.6; % reference friction coefficient
M.V0 = 1e-6; % reference slip velocity
M.a = 0.01; % direct effect parameter
M.b = 0.015; % state evolution parameter
M.dc = 10e-6; % (m) state evolution distance

% elasticity parameters
rho = 2.7; % density (g/cm^3)
c = 3.5; % S-wave speed (km/s)
M.eta = rho*c/2; % radiation-damping coefficient (MPa*s/m)

% effective normal stress
M.N = 50; % (MPa)

% critical stiffness (neglecting radiation-damping)
kcr = M.N*(M.b-M.a)/M.dc; % (MPa/m)

% stiffness
M.k = 0.9*kcr; % (MPa/m) k<kcr gives stick-slip

% stressing rate (shear stress increases at constant rate in absence of slip)
M.dtaudt = 1e-5; % (MPa/s)

% initial conditions
D0 = 0; % slip (m)
M.tau0 = M.f0*M.N; % shear stress (MPa)
Psi0 = M.f0; % state variable

% time
tmax = 2e6;

% constant time step solution
solveConstant = false;

if solveConstant
    
    nt = 100000;
    t = linspace(0,tmax,nt+1)'; dt = tmax/nt;
    
    % store solution, including initial condition
    D = nan(nt+1,1);
    Psi = nan(nt+1,1);
    tau = nan(nt+1,1);
    V = nan(nt+1,1);
    
    D(1)=D0; Psi(1)=Psi0;
    for n=1:nt
        
        % note: V = dD/dt, G = dPsi/dt
        
        t0 = t(n); % time at start of time step
        
        % fourth order RK, numbers refer to RK stage
        
        [V1,G1,tau1] = sliderODE(D(n),Psi(n),t0,M);
        
        V(n)=V1; tau(n)=tau1; % stage 1 values are stored
        
        [V2,G2] = sliderODE(D(n)+0.5*dt*V1,Psi(n)+0.5*dt*G1,t0+0.5*dt,M);
        [V3,G3] = sliderODE(D(n)+0.5*dt*V2,Psi(n)+0.5*dt*G2,t0+0.5*dt,M);
        [V4,G4] = sliderODE(D(n)+    dt*V3,Psi(n)+    dt*G3,t0+    dt,M);
        
        D(n+1)   = D(n)  +dt/6*(V1+2*V2+2*V3+V4);
        Psi(n+1) = Psi(n)+dt/6*(G1+2*G2+2*G3+G4);
        
        semilogy(t,V),drawnow
        
    end
    
    figure(1),clf
    subplot(3,1,1)
    semilogy(t,V)
    xlabel('time')
    ylabel('slip velocity (m/s)')
    title(['dt = ',num2str(dt)])
    hold on
    
end

% now repeat using adaptive time-stepping

solveAdaptive = true;
dispAcceptReject = false;
plotSolDuringSim = true;

if solveAdaptive

    tol = 1e-4; % tolerance
    dt = 1; % initial time step (s)
    dtmax = 1e5; % maximum time step (s)
    safety = 0.9; % safety factor
    
    % ta = time vector (a=adaptive)
    % Da,Psia,Va,taua = solution (a=adaptive)
    
    t=0;
    
    % store solution
    ta=t; Da=D0; Psia=Psi0;
    [V1,G1,tau1] = sliderODE(Da(end),Psia(end),t,M);
    Va=V1; taua=tau1; % stage 1 values are stored
    
    err=0; dta=dt;
    
    while t<tmax
        
        % adjust dt to stop at tmax
        if t+dt>tmax, dt=tmax-t; end
        
        % three-stage method with embedded error estimate
        
        [V2,G2] = sliderODE(Da(end)+0.5*dt*V1,Psia(end)+0.5*dt*G1,t+0.5*dt,M);
        [V3,G3] = sliderODE(Da(end)+dt*(-V1+2*V2),Psia(end)+dt*(-G1+2*G2),t+dt,M);
        
        % second order update
        D2   = Da(end)  +dt/2*(V1+V3);
        Psi2 = Psia(end)+dt/2*(G1+G3);
        
        % third order update
        D3   = Da(end)  +dt/6*(V1+4*V2+V3);
        Psi3 = Psia(end)+dt/6*(G1+4*G2+G3);
        
        q = 2; % order of accuracy of lower order update
        
        % local error estimate
        er = norm([D2-D3; Psi2-Psi3]);
        
        if er<tol
            % update solution
            t = t+dt; ta=[ta; t];
            Da = [Da; D3]; Psia = [Psia; Psi3]; % use third-order update
            
            % store error and time step
            err=[err; er]; dta=[dta; dt];
            
            % evaluate stage 1 values for next time step
            [V1,G1,tau1] = sliderODE(Da(end),Psia(end),t,M);
            Va=[Va; V1]; taua=[taua; tau1]; % stage 1 values are stored

            if dispAcceptReject, disp(['accept ' num2str(t) ' ' num2str(er) ' ' num2str(dt)]), end
            if plotSolDuringSim, semilogy(ta,Va), drawnow, end
            
        else
            if dispAcceptReject, disp(['reject ' num2str(t) ' ' num2str(er) ' ' num2str(dt)]), end
        end
        
        % adjust time step
        dt = safety*dt*(tol/er)^(1/(q+1));
        dt = min(dt,dtmax);
        
    end

    % plot
    subplot(3,1,1)
    semilogy(ta,Va)
    
    subplot(3,1,2)
    plot(ta,err,[0 tmax],[tol tol],'k--')
    xlabel('time')
    ylabel('error')
    
    subplot(3,1,3)
    semilogy(1:length(ta),dta)
    xlabel('time step number')
    ylabel('\Deltat')

end
    
% functions below here

function [V,G,tau] = sliderODE(D,Psi,t,M)
    
    % evaluate stress when V=0

    tauLock = M.tau0+M.dtaudt*t-M.k*D;

    % set bounds on V for root-finding

    if tauLock>0
        Vmin = 0; Vmax = tauLock/M.eta;
    else
        Vmin = tauLock/M.eta; Vmax = 0;
    end

    % solve stress=strength for V
    
    atol = 1e-14; rtol = 1e-6;
    
    V = hybrid(@(V) solveV(V,tauLock,Psi,M) ,Vmin,Vmax,atol,rtol);
    
    % then evaluate tau

    tau = tauLock-M.eta*V;

    % and state evolution, G = dPsi/dt

    if V==0
        G = 0; % special case to avoid log(0)
    else
        f = tau/M.N; fss = M.f0+(M.a-M.b)*log(V/M.V0);
        G = -V/M.dc*(f-fss);
    end
        
end


function residual = solveV(V,tauLock,Psi,M)

    stress = tauLock-M.eta*V;
    %f = M.a*log(V/M.V0)+Psi;
    f = M.a*asinh(V/(2*M.V0)*exp(Psi/M.a));
    strength = f*M.N;
    residual = stress-strength;
    
end


function [x,err]=hybrid(func,a,b,atol,rtol)

  % hybrid method solves func(x)=0 for some root x within (a,b)
  % returns x, estimate of root with absolute error less than atol
  % or relative error less than rtol
  
  % function values at endpoints
  fa = func(a);
  fb = func(b);

  % make sure root is bracketed; otherwise return
  if sign(fa)==sign(fb) | isnan(fa) | isnan(fb)
    disp('error: root not bracketed or function is NaN at endpoint')
    x = NaN; err = NaN;
    return
  end

  % set up secant method, storing old values as xold and fold, new ones as x and f
  % use bisection brackets to start secant (this is somewhat arbitrary)
  xold = a;
  fold = fa;
  x = b;
  f = fb;
  
  % begin iterations,
  % keeping track of error at each iteration in vector err
  n = 0; err = [];
  update = 'input'; % character string stating type of update used in previous interation
  while b-a>atol+rtol*abs(x) % safe to have infinite loop since bisection guaranteed to converge
      
      err = [err b-a]; % add to end of vector the current error (interval width)

      % formatted printing so you can watch method converge
      %fprintf('%6i %20.10f %20.10f %20.10f %s\n',n,a,x,b,update)

      n = n+1; % iteration number

      % first calculate (tenative) secant update
      dfdx = (f-fold)/(x-xold); % approximation to df/dx
      dx = -f/dfdx; % update to x
      xs = x+dx; % secant update
      
      % determine if secant method will be used
      if (xs<a) | (xs>b)
          use_secant = false;  % not if update outside (a,b)
      else
          fs = func(xs); % function value at secant update
          % calculate interval reduction factor = (old interval width)/(new interval width)
          if sign(fs)==sign(fa)
              IRF = (b-a)/(b-xs); % would update a=xs
          else
              IRF = (b-a)/(xs-a); % would update b=xs
          end
          if IRF<2
              use_secant = false;
          else
              use_secant = true;
          end
      end

      xold = x; fold = f; % store these values for next iteration

      % now update
      if use_secant
          update = 'secant';
          x = xs;
          f = fs;
      else
          update = 'bisection';
          x = (a+b)/2; % midpoint
          f = func(x); % function value at midpoint
      end
      
      % update one endpoint based on sign of function value at updated x
      if sign(f)==sign(fa)
          a = x;
      else
          b = x;
      end
      
  end
  
end

